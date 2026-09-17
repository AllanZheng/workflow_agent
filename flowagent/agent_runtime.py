"""受管 Agent 节点的请求级 MCP 工具循环。"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


class ToolIterationLimit(RuntimeError):
    pass


class ManagedAgentRuntime:
    def __init__(self, llm, mcp_client) -> None:
        self.llm = llm
        self.mcp_client = mcp_client

    async def run(self, message: str, system_prompt: str,
                  max_tool_iterations: int) -> str:
        remote_tools = await self.mcp_client.list_tools()
        allowed = {tool["name"] for tool in remote_tools}
        schemas = [{
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema") or {"type": "object"},
            },
        } for tool in remote_tools]
        model = self.llm.bind_tools(schemas)
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=message))

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
