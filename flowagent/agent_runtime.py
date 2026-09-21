"""受管 Agent 节点的请求级 MCP 工具循环。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


class ToolIterationLimit(RuntimeError):
    pass


class ManagedAgentRuntime:
    def __init__(self, llm, mcp_client=None) -> None:
        self.llm = llm
        self.mcp_client = mcp_client

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
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        for item in history or []:
            role = item.get("role")
            content = str(item.get("content", ""))
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=message))
        return allowed, schemas, messages

    async def run(self, message: str, system_prompt: str,
                  max_tool_iterations: int,
                  history: list[dict] | None = None) -> str:
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
                name = call["name"]
                if name not in allowed:
                    raise ValueError(f"tool_not_allowed: {name}")
                result = await self.mcp_client.call_tool(name, call.get("args") or {})
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        raise ToolIterationLimit("managed agent tool iteration limit exceeded")

    async def stream(self, message: str, system_prompt: str,
                     max_tool_iterations: int,
                     history: list[dict] | None = None):
        """流式执行 Agent，且只产出最终无工具调用轮次的文本内容。

        工具调用仍在本地循环完成；工具名、参数和结果均不写入调用方的文本流。
        """
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
                name = call["name"]
                if name not in allowed:
                    raise ValueError(f"tool_not_allowed: {name}")
                result = await self.mcp_client.call_tool(name, call.get("args") or {})
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        raise ToolIterationLimit("managed agent tool iteration limit exceeded")
