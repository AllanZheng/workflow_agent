"""受管 Agent 节点的请求级 MCP 工具循环。"""

from __future__ import annotations

import inspect
import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from .mcp_edge_client import McpEdgeError
from .skill_runtime import SkillLoadError


class ToolIterationLimit(RuntimeError):
    pass


@dataclass(frozen=True)
class _GroupSelectionRequired:
    message: str


# 本地终端工具的结果：文本直接作为最终回复，不再交回模型改写。
# English: Result of a local terminal tool; its text becomes the final reply verbatim.
@dataclass(frozen=True)
class _TerminalReply:
    text: str
    metadata: dict


# 本地工具：由运行时自己执行，而不是转发到 MCP。requires_remote 指定它依赖的远端工具，
# 远端工具对当前执行不可见时本地工具也不曝露。
# English: A runtime-local tool; hidden when its required remote tool is not visible.
@dataclass(frozen=True)
class LocalTool:
    name: str
    description: str
    parameters: dict
    handler: Callable[[dict], Awaitable[tuple[str, dict]]]
    requires_remote: str | None = None
    terminal: bool = True


@dataclass
class RuntimeUsage:
    loaded_skills: list[dict[str, str]] = field(default_factory=list)
    skill_load_failures: list[dict[str, str]] = field(default_factory=list)
    called_tools: list[dict[str, str]] = field(default_factory=list)
    # 仅在 record_tool_events=True 时填充：每次远端工具调用的远端名、结果与载荷。
    tool_events: list[dict[str, Any]] = field(default_factory=list)


class ManagedAgentRuntime:
    # 初始化对象状态并建立所需依赖。
    # English: Initialize object state and required dependencies.
    def __init__(self, llm, mcp_client=None, *, skills=None, skill_loader=None,
                 execution_id: str | None = None, generation: int | None = None,
                 local_tools=None, tolerate_tool_errors: bool = False,
                 record_tool_events: bool = False) -> None:
        self.llm = llm
        self.mcp_client = mcp_client
        self.loaded_skills: list[dict] = []
        self.skills = list(skills or [])
        self.skill_loader = skill_loader
        self.execution_id = execution_id
        self.generation = generation
        self.usage = RuntimeUsage()
        self.local_tools: list[LocalTool] = list(local_tools or [])
        # 为真时工具报错不中断循环，而是以错误文本交回模型（函数阶段需要据此统计失败）。
        self.tolerate_tool_errors = tolerate_tool_errors
        self.record_tool_events = record_tool_events
        self.terminal_metadata: dict = {}
        self._active_local: dict[str, LocalTool] = {}
        self._skills_by_id = {str(skill["skill_id"]): skill for skill in self.skills}
        if len(self._skills_by_id) != len(self.skills):
            raise ValueError("duplicate_skill_binding")
        self._loaded_skill_bodies: dict[str, str] = {}
        self._dependency_checked_skills: set[str] = set()

    # 记录技能加载。
    # English: Record skill load.
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
    # English: Handle setup operations.
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
        # 本地工具只在其依赖的远端工具对本次执行可见时才注册。
        self._active_local = {}
        for local in self.local_tools:
            if local.requires_remote and not any(
                    tool.get("remote_name") == local.requires_remote for tool in remote_tools):
                continue
            if local.name in allowed:
                raise ValueError(f"reserved_tool_name: {local.name}")
            allowed.add(local.name)
            self._active_local[local.name] = local
            schemas.append({
                "type": "function",
                "function": {"name": local.name, "description": local.description,
                             "parameters": local.parameters},
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
    # English: Handle system prompt with skill summaries operations.
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
    # English: Load skill.
    async def _load_skill(self, skill_id: str) -> str:
        skill = self._skills_by_id.get(skill_id)
        if skill is None:
            raise ValueError("skill_not_bound")
        if skill_id in self._loaded_skill_bodies and skill_id not in self._dependency_checked_skills:
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
        source_body_hash = body_hash
        if isinstance(result, dict) and result.get("requires_dependency_validation"):
            self._dependency_checked_skills.add(skill_id)
            source_body_hash = result["source_body_hash"]
        record = {
            "skill_id": skill_id,
            "metadata_version_id": skill["metadata_version_id"],
            "body_version_id": skill["body_version_id"],
            "body_hash": source_body_hash,
            "outcome": "success",
        }
        if source_body_hash != body_hash:
            record["resolved_body_hash"] = body_hash
        self.usage.loaded_skills.append(record)
        return body

    # 处理`call`工具相关逻辑。
    # English: Handle call tool operations.
    async def _call_tool(self, call: dict, allowed: set[str]) -> str | _GroupSelectionRequired | _TerminalReply:
        name = call["name"]
        if name not in allowed:
            raise ValueError(f"tool_not_allowed: {name}")
        arguments = call.get("args") or {}
        local = self._active_local.get(name)
        if local is not None:
            text, metadata = await local.handler(dict(arguments))
            return _TerminalReply(text, metadata)
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
        remote_name = descriptor.get("remote_name") if descriptor is not None else name
        try:
            result = await self.mcp_client.call_tool(name, arguments)
        except McpEdgeError as exc:
            if not self.tolerate_tool_errors:
                raise
            if self.record_tool_events:
                self.usage.tool_events.append(
                    {"remote_name": remote_name, "outcome": "failed", "payload": None})
            return f"Tool call failed: {exc}"
        if self.record_tool_events:
            self.usage.tool_events.append(
                {"remote_name": remote_name, "outcome": "success", "payload": result})
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
    # English: Handle skill load error operations.
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
    # English: Run the current operation.
    async def run(self, message: str, system_prompt: str,
                  max_tool_iterations: int,
                  history: list[dict] | None = None) -> str:
        self.loaded_skills = []
        self.usage = RuntimeUsage()
        self._loaded_skill_bodies = {}
        self.terminal_metadata = {}
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
                if isinstance(result, _TerminalReply):
                    self.terminal_metadata = result.metadata
                    return result.text
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        raise ToolIterationLimit("managed agent tool iteration limit exceeded")

    # English: Handle stream operations.
    async def stream(self, message: str, system_prompt: str,
                     max_tool_iterations: int,
                     history: list[dict] | None = None):
        """流式执行 Agent，且只产出最终无工具调用轮次的文本内容。

        工具调用仍在本地循环完成；工具名、参数和结果均不写入调用方的文本流。
        """
        self.loaded_skills = []
        self.usage = RuntimeUsage()
        self._loaded_skill_bodies = {}
        self.terminal_metadata = {}
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
                if isinstance(result, _TerminalReply):
                    # 终端工具结果一次性作为唯一的文本增量，不交回模型改写。
                    self.terminal_metadata = result.metadata
                    yield result.text
                    return
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        raise ToolIterationLimit("managed agent tool iteration limit exceeded")
