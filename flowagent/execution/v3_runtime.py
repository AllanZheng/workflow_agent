"""V3 对话与功能阶段共用的受管运行时构造。"""

from ..agent_runtime import ManagedAgentRuntime
from ..execution_authorization import AuthorizedChatModel, DemoModelAuthorizer
from ..local_tools import build_group_analysis_tool
from ..mcp_edge_client import McpEdgeClient
from ..skill_runtime import DemoSkillBodyLoader


def execution_parts(request, model_authorizer, skill_loader, llm_factory):
    # 保留每次执行的快照、模型授权和工具绑定，流式入口也走同一逻辑。
    config = request.agent_config
    snapshot = config.snapshot_v3
    authorizer = model_authorizer or DemoModelAuthorizer(
        str(config.model_authorization_url), config.execution_token.get_secret_value(),
        str(snapshot.runtime_id), credential_generation=config.model_credential_generation,
    )
    llm = AuthorizedChatModel(
        llm_factory(None, config), authorizer, execution_id=request.execution_id,
        model_version_id=str(snapshot.model.version_id), generation=request.binding_generation,
    )
    mcp_client = None
    if snapshot.tools:
        mcp_client = McpEdgeClient(
            str(config.mcp_endpoint), config.execution_token.get_secret_value(),
            fixed_tools=[tool.model_dump(mode="json") for tool in snapshot.tools],
            execution_id=request.execution_id, agent_id=str(snapshot.agent_id),
            revision_id=str(snapshot.revision_id), runtime_id=str(snapshot.runtime_id),
            generation=request.binding_generation,
        )
    resolved_skill_loader = skill_loader
    if snapshot.skills and resolved_skill_loader is None:
        resolved_skill_loader = DemoSkillBodyLoader(
            str(config.model_authorization_url), config.execution_token.get_secret_value(),
            str(snapshot.runtime_id),
            fixed_tools=[tool.model_dump(mode="json") for tool in snapshot.tools],
        )
    return snapshot, llm, mcp_client, resolved_skill_loader


def chat_local_tools(request, snapshot, llm, mcp_client):
    # 群分析终端工具仅在快照已授权读取群消息时可用。
    if mcp_client is None or not any(
            tool.remote_name == "fetch_group_analysis_input" for tool in snapshot.tools):
        return []
    return [build_group_analysis_tool(llm, mcp_client, request.analysis_target)]


def chat_runtime(request, snapshot, llm, mcp_client, skill_loader):
    # 工具返回错误时将错误交给模型修正参数；仍受工具白名单和八轮上限约束。
    return ManagedAgentRuntime(
        llm, mcp_client, skills=[skill.model_dump(mode="json") for skill in snapshot.skills],
        skill_loader=skill_loader, execution_id=request.execution_id,
        generation=request.binding_generation, tolerate_tool_errors=True,
        local_tools=chat_local_tools(request, snapshot, llm, mcp_client),
    )
