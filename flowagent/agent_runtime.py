"""受管 Agent 节点的请求级 MCP 工具循环。"""

from __future__ import annotations

import inspect
import hashlib
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from .skill_runtime import SkillLoadError


class ToolIterationLimit(RuntimeError):
    pass


@dataclass(frozen=True)
class _GroupSelectionRequired:
    message: str


@dataclass
class RuntimeUsage:
    loaded_skills: list[dict[str, str]] = field(default_factory=list)
    skill_load_failures: list[dict[str, str]] = field(default_factory=list)
    called_tools: list[dict[str, str]] = field(default_factory=list)


class ManagedAgentRuntime:
    # 初始化对象状态并建立所需依赖。
    def __init__(self, llm, mcp_client=None, *, skills=None, skill_loader=None,
                 execution_id: str | None = None, generation: int | None = None) -> None:
        self.llm = llm
        self.mcp_client = mcp_client
        self.loaded_skills: list[dict] = []
        self.skills = list(skills or [])
        self.skill_loader = skill_loader
        self.execution_id = execution_id
        self.generation = generation
        self.usage = RuntimeUsage()
        self._skills_by_id = {str(skill["skill_id"]): skill for skill in self.skills}
        if len(self._skills_by_id) != len(self.skills):
            raise ValueError("duplicate_skill_binding")
        self._loaded_skill_bodies: dict[str, str] = {}

    # 记录技能加载。
    def _record_skill_load(self, tool_name: str, result) -> None:
        if tool_name != "platform_load_skill" or not isinstance(result, dict):
            return
        details = result.get("result")
        if not isinstance(details, dict):
            return
        loaded = {
            "skill_key": details.get("skill_key"),
            "metadata_version_id": details.get("metadata_version_id"),
            "body_version_id": details.get("body_version_id"),
            "content_hash": details.get("content_hash"),
        }
        if all(isinstance(value, str) and value for value in loaded.values()):
            self.loaded_skills.append(loaded)

    # 处理`setup`相关逻辑。
    async def _setup(self, message: str, system_prompt: str, history: list[dict] | None):
        # 受管 Agent 可显式禁用工具：纯文本工作流不应为了取得一个空列表
        # 而依赖 MCP 服务。调用方负责拒绝半配置的 MCP（endpoint/token 仅其一）。
        remote_tools = await self.mcp_client.list_tools() if self.mcp_client is not None else []
        allowed = {tool["name"] for tool in remote_tools}
        schemas = [{
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema") or {"type": "object"},
            },
        } for tool in remote_tools]
        if self.skills:
            if "load_skill" in allowed:
                raise ValueError("reserved_tool_name: load_skill")
            allowed.add("load_skill")
            schemas.append({
                "type": "function",
                "function": {
                    "name": "load_skill",
                    "description": "Load the fixed body of a published Skill when its summary is not enough.",
                    "parameters": {
                        "type": "object",
                        "properties": {"skill_id": {
                            "type": "string", "enum": sorted(self._skills_by_id),
                        }},
                        "required": ["skill_id"],
                        "additionalProperties": False,
                    },
                },
            })
        messages = []
        full_system_prompt = self._system_prompt_with_skill_summaries(system_prompt)
        if full_system_prompt:
            messages.append(SystemMessage(content=full_system_prompt))
        for item in history or []:
            role = item.get("role")
            content = str(item.get("content", ""))
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=message))
        return allowed, schemas, messages

    # 处理`system`提示词包含技能摘要相关逻辑。
    def _system_prompt_with_skill_summaries(self, system_prompt: str) -> str:
        sections = [system_prompt.strip()] if system_prompt.strip() else []
        if self.skills:
            summary_lines = ["Published Skill summaries:"]
            for skill in self.skills:
                summary_lines.append(
                    f"- {skill['name']} (skill_id={skill['skill_id']}): {skill['summary']}"
                )
            sections.append("\n".join(summary_lines))
        return "\n\n".join(sections)

    # 加载技能。
    async def _load_skill(self, skill_id: str) -> str:
        skill = self._skills_by_id.get(skill_id)
        if skill is None:
            raise ValueError("skill_not_bound")
        if skill_id in self._loaded_skill_bodies:
            return self._loaded_skill_bodies[skill_id]
        if self.skill_loader is None:
            raise ValueError("skill_loader_unavailable")
        if not self.execution_id:
            raise ValueError("skill_execution_scope_missing")
        result = self.skill_loader(
            execution_id=self.execution_id, generation=self.generation, skill=skill,
        )
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, dict):
            if result.get("skill_id", skill_id) != skill_id:
                raise ValueError("skill_binding_mismatch")
            if result.get("metadata_version_id", skill["metadata_version_id"]) != skill["metadata_version_id"]:
                raise ValueError("skill_metadata_version_mismatch")
            if result.get("body_version_id") != skill["body_version_id"]:
                raise ValueError("skill_body_version_mismatch")
            body = result.get("body_text")
            returned_body_hash = result.get("body_hash")
        else:
            # Injected loaders may return the body directly; the production HTTP loader
            # validates IDs and content hash before returning it.
            body = result
        if not isinstance(body, str):
            raise ValueError("skill_body_invalid")
        body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if isinstance(result, dict) and returned_body_hash != body_hash:
            raise ValueError("skill_body_hash_mismatch")
        self._loaded_skill_bodies[skill_id] = body
        self.usage.loaded_skills.append({
            "skill_id": skill_id,
            "metadata_version_id": skill["metadata_version_id"],
            "body_version_id": skill["body_version_id"],
            "body_hash": body_hash,
            "outcome": "success",
        })
        return body

    # 处理`call`工具相关逻辑。
    async def _call_tool(self, call: dict, allowed: set[str]) -> str | _GroupSelectionRequired:
        name = call["name"]
        if name not in allowed:
            raise ValueError(f"tool_not_allowed: {name}")
        arguments = call.get("args") or {}
        if name == "load_skill":
            skill_id = str(arguments.get("skill_id", ""))
            try:
                return await self._load_skill(skill_id)
            except SkillLoadError as exc:
                return self._skill_load_error(skill_id, str(exc))
            except ValueError as exc:
                return self._skill_load_error(skill_id, str(exc))
            except Exception:
                return self._skill_load_error(skill_id, "skill_load_failed")
        descriptor = getattr(self.mcp_client, "tool_bindings", {}).get(name)
        result = await self.mcp_client.call_tool(name, arguments)
        if name == "platform_resolve_group" and isinstance(result, dict):
            if result.get("status") == "not_found":
                return _GroupSelectionRequired("没有找到你可访问的目标群，请确认群名或提供群 ID。")
            if result.get("status") == "ambiguous":
                choices = "、".join(
                    f"{item['name']}（ID {item['conversation_id']}）"
                    for item in result.get("candidates", [])
                    if isinstance(item, dict) and isinstance(item.get("name"), str)
                    and type(item.get("conversation_id")) is int
                )
                return _GroupSelectionRequired(f"找到多个同名或相近的群：{choices}。请说明要分析的群 ID。")
        self._record_skill_load(name, result)
        if descriptor is not None:
            self.usage.called_tools.append({
                "tool_id": str(descriptor["tool_id"]),
                "tool_version_id": str(descriptor["tool_version_id"]),
                "server_version_id": str(descriptor["server_version_id"]),
                "outcome": "success",
            })
        return str(result)

    # 处理技能加载错误相关逻辑。
    def _skill_load_error(self, skill_id: str, error_code: str) -> str:
        skill = self._skills_by_id.get(skill_id)
        if skill is not None:
            self.usage.skill_load_failures.append({
                "skill_id": skill_id,
                "metadata_version_id": str(skill["metadata_version_id"]),
                "body_version_id": str(skill["body_version_id"]),
                "outcome": "failed",
                "error_code": error_code if error_code.startswith(("skill_", "execution_")) else "skill_load_failed",
            })
            return "The pinned Skill body could not be loaded; continue without its instructions."
        return "That Skill is not bound to this execution; continue without it."

    # 执行当前任务流程并返回运行结果。
    async def run(self, message: str, system_prompt: str,
                  max_tool_iterations: int,
                  history: list[dict] | None = None) -> str:
        self.loaded_skills = []
        self.usage = RuntimeUsage()
        self._loaded_skill_bodies = {}
        allowed, schemas, messages = await self._setup(message, system_prompt, history)
        model = self.llm.bind_tools(schemas)

        for iteration in range(max_tool_iterations + 1):
            response = await model.ainvoke(messages)
            tool_calls = list(getattr(response, "tool_calls", None) or [])
            if not tool_calls:
                return str(response.content)
            if iteration >= max_tool_iterations:
                raise ToolIterationLimit("managed agent tool iteration limit exceeded")
            messages.append(response)
            for call in tool_calls:
                result = await self._call_tool(call, allowed)
                if isinstance(result, _GroupSelectionRequired):
                    return result.message
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        raise ToolIterationLimit("managed agent tool iteration limit exceeded")

    async def stream(self, message: str, system_prompt: str,
                     max_tool_iterations: int,
                     history: list[dict] | None = None):
        """流式执行 Agent，且只产出最终无工具调用轮次的文本内容。

        工具调用仍在本地循环完成；工具名、参数和结果均不写入调用方的文本流。
        """
        self.loaded_skills = []
        self.usage = RuntimeUsage()
        self._loaded_skill_bodies = {}
        allowed, schemas, messages = await self._setup(message, system_prompt, history)
        model = self.llm.bind_tools(schemas)

        for iteration in range(max_tool_iterations + 1):
            response = None
            text_chunks: list[str] = []
            async for chunk in model.astream(messages):
                response = chunk if response is None else response + chunk
                content = getattr(chunk, "content", "")
                # 空工具路径下可立即透传，获得真正 token 级流式。若本轮工具
                # 可用则先暂存，直到确认它不是工具调用轮次，避免泄露中间推理。
                if isinstance(content, str) and content:
                    if schemas:
                        text_chunks.append(content)
                    else:
                        yield content
            if response is None:
                response = AIMessage(content="")
            tool_calls = list(getattr(response, "tool_calls", None) or [])
            if not tool_calls:
                for content in text_chunks:
                    yield content
                return
            if iteration >= max_tool_iterations:
                raise ToolIterationLimit("managed agent tool iteration limit exceeded")
            messages.append(response)
            for call in tool_calls:
                result = await self._call_tool(call, allowed)
                if isinstance(result, _GroupSelectionRequired):
                    yield result.message
                    return
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        raise ToolIterationLimit("managed agent tool iteration limit exceeded")
