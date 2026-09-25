# 群分析助手 MVP · 发布与联调手册（阶段 3）

> 交付物：im-tools-mcp 服务（本仓库 `services/im_tools_mcp/`）+ 受管工作流
> `configs/managed/group_analysis.yaml` + 本目录的 compose 接入与 Admin 发布脚本。
> 当前流程以 IM `conversation_id` 为唯一目标。调用方可显式传 `analysis_target`；未传时 flow_agent 从消息提取群名，平台按当前用户的 IM 群成员关系解析并绑定唯一 ID。

## 1. 架构与部署面

```
Demo 前端 ── platform(demo_platform, agent-mvp 栈) ── workflow(execution-v1)
                │ MCP Edge /mcp/runtime/<exec>/（按 execution 授权）
                ▼
          im-tools-mcp(本仓库, :8090) ── im(既有部署, REST)
```

部署需更新 **im-tools-mcp**、workflow-agent 和 demo-platform，并先执行 demo-platform 的 `r007_group_analysis_results` 数据库迁移。

## 2. 起栈（在 demo_platform 目录）

```bash
cd demo_platform
# .env 需含 agent-mvp 栈既有变量 + 以下新增：
#   IM_BASE_URL=http://<im 地址>            # 如 http://host.docker.internal:8000
#   IM_TOOLS_MCP_TOKEN=<im 超管令牌>        # 与下方 IM_ACCESS_TOKEN 同值（单令牌双用）
docker compose -f docker-compose.agent-mvp.yml \
  -f ../workflow_agent/deploy/group-analysis/compose.group-analysis.yml \
  --env-file .env up -d --build
```

验证：`curl http://服务器:8091/health` → `{"ok":true}`。

## 3. Admin 发布（写入智能体「群分析助手」）

发布脚本（零改 admin 代码，直接写 admin 配置库）：

先在 Admin 的 MCP 服务管理中登记 `im-tools-mcp`，地址须与下方
`GROUP_ANALYSIS_IM_TOOLS_URL` 完全一致；执行“发现工具”，确认列表中有
`fetch_group_analysis_input`，再启用该服务和工具。记录服务 ID、工具 ID，分别填入
`GROUP_ANALYSIS_MCP_SERVER_ID`、`GROUP_ANALYSIS_INPUT_TOOL_ID`。这两个 ID 是
环境内实际注册值，不能猜测或使用示例值。发布脚本会在写修订前核对注册记录。

```bash
cd workflow_agent
export AGENT_PLATFORM_DATABASE_URL=<admin 库连接串>
export AGENT_CONFIG_MASTER_KEY_B64=<与 platform 一致的 32B 主密钥>
export GROUP_ANALYSIS_AGENT_ID=$(uuidgen)                 # 或既有 workflow 智能体 UUID（接管）
export GROUP_ANALYSIS_MODEL_CREDENTIAL_ID=$(uuidgen)
export GROUP_ANALYSIS_IM_CREDENTIAL_ID=$(uuidgen)
export GROUP_ANALYSIS_MODEL_NAME=qwen3.5-35b
export GROUP_ANALYSIS_MODEL_BASE_URL=http://192.168.1.200:9785/v1
export GROUP_ANALYSIS_MODEL_API_KEY=<vLLM key>
export GROUP_ANALYSIS_IM_ACCESS_TOKEN=<im 超管令牌>       # mcp_headers 凭据内容
export GROUP_ANALYSIS_IM_TOOLS_URL=http://im-tools-mcp:8090/mcp/
export GROUP_ANALYSIS_MCP_SERVER_ID=<已启用的 im-tools-mcp 服务 ID>
export GROUP_ANALYSIS_INPUT_TOOL_ID=<已注册 fetch_group_analysis_input 工具 ID>
python deploy/group-analysis/publish_group_analysis.py --dry-run
python deploy/group-analysis/publish_group_analysis.py
```

脚本写入（幂等，同 digest 不产生新修订）：

| 写入项 | 内容 |
| --- | --- |
| credentials(model_api_key) | vLLM key，AES-256-GCM（Admin 契约） |
| credentials(mcp_headers) | `{"Authorization": "Bearer <im 令牌>"}`，MCP Edge 按请求注入 |
| agents + agent_config_revisions | 快照：group_analysis.yaml + 模型 + 已注册的 MCP 工具引用（schema 与服务实现同源） |
| agent_platform_settings | 设为默认智能体（可 `GROUP_ANALYSIS_SET_AS_DEFAULT=false` 关闭） |

**改口径/改流程**：改 `configs/managed/group_analysis.yaml` → 重跑发布脚本
（新 digest → 新修订）；对话级临时口径由用户在会话中说「按 XX 分类」即可。

## 4. Demo 前端手工联调（验收清单）

1. 在 AIChat 发送「分析验收测试群最近 7 天」，或从群选择器取得 IM `conversation_id` 并在聊天请求增加 `"analysis_target":{"conversation_id":5}`：
   - 期望：流式输出 Markdown 报告，含 `## 总结` / `## 分类统计` / `## 汇报要点`；
2. 未提供群名和 `analysis_target` → 提示补充群名；同名群 → 列出当前用户可读的候选群及 ID，选择后继续。显式传入的目标 ID 不受消息中其他群名影响。
3. 指定时间段（如「9月1日到9月15日」）→ 结束日期当天纳入，次日零点排除。
4. 30 天大窗口 → 结果注明完整或部分完成；不能把未知数据写成完整覆盖。
5. im 停机模拟 → 用户可读错误提示，会话不崩，im 恢复后可重试。
6. PostgreSQL 先升级到 `r007_group_analysis_results`；完成后按 `GET /api/v1/agent/group-analysis/results?conversation_id=5` 查询本人仍有权限查看的历史分析。

## 5. 已知边界（记录在案）

- 受管执行要求模型 key（引擎语义）。「无 key Mock 演示」用 stub OpenAI 兼容端点
  （`GROUP_ANALYSIS_MODEL_BASE_URL` 指 stub）实现，无需改引擎。
- 分析输入由程序一次性按 ID 分页读取并去重；模型只处理拿到的消息。默认最多 31 天、100 页、1000 条及 120000 字符，触限显示部分完成。
- 请求用户须有有效的 IM 群成员映射；服务凭据仅用于调用 IM，不代替用户授权。
- IM 令牌须为 IM 接受的 ca1 凭据；admin-api 的短期登录令牌不能直接作为此处静态凭据。
