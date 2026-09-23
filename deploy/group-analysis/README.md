# 群分析助手 MVP · 发布与联调手册（阶段 3）

> 交付物：im-tools-mcp 服务（本仓库 `services/im_tools_mcp/`）+ 受管工作流
> `configs/managed/group_analysis.yaml` + 本目录的 compose 接入与 Admin 发布脚本。
> 约束：im / demo_platform / admin 代码零改动；引擎（flowagent/）零改动。

## 1. 架构与部署面

```
Demo 前端 ── platform(demo_platform, agent-mvp 栈) ── workflow(execution-v1)
                │ MCP Edge /mcp/runtime/<exec>/（按 execution 授权）
                ▼
          im-tools-mcp(本仓库, :8090) ── im(既有部署, REST)
```

部署面唯一新增单元：**im-tools-mcp** 服务（compose 服务项 + 镜像）。

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
python deploy/group-analysis/publish_group_analysis.py
```

脚本写入（幂等，同 digest 不产生新修订）：

| 写入项 | 内容 |
| --- | --- |
| credentials(model_api_key) | vLLM key，AES-256-GCM（Admin 契约） |
| credentials(mcp_headers) | `{"Authorization": "Bearer <im 令牌>"}`，MCP Edge 按请求注入 |
| agents + agent_config_revisions | 快照：group_analysis.yaml + 模型 + mcp_tools 注册（schema 与服务实现同源） |
| agent_platform_settings | 设为默认智能体（可 `GROUP_ANALYSIS_SET_AS_DEFAULT=false` 关闭） |

**改口径/改流程**：改 `configs/managed/group_analysis.yaml` → 重跑发布脚本
（新 digest → 新修订）；对话级临时口径由用户在会话中说「按 XX 分类」即可。

## 4. Demo 前端手工联调（验收清单）

1. 打开 Demo 前端（platform SPA），会话内发送「分析<群名>最近 7 天」：
   - 期望：流式输出 Markdown 报告，含 `## 总结` / `## 分类统计` / `## 汇报要点`；
2. 缺群名（如「帮我分析一下群」）→ 期望反问；补充群名后再发 → 成功出报告；
3. 指定时间段（如「分析 X 群，9月1日到9月15日」）→ 报告时间窗正确；
4. 30 天大窗口 → 报告带覆盖率声明（分段汇总完成）；
5. im 停机模拟 → 用户可读错误提示，会话不崩，im 恢复后可重试。

## 5. 已知边界（记录在案）

- 受管执行要求模型 key（引擎语义）。「无 key Mock 演示」用 stub OpenAI 兼容端点
  （`GROUP_ANALYSIS_MODEL_BASE_URL` 指 stub）实现，无需改引擎。
- `max_tool_iterations` 引擎护栏 1..8：靠 agent 每轮**并行多段**工具调用覆盖
  30 天窗口（fetch_sys 已写明该指令）；采样兜底 + 覆盖率声明见 YAML。
- im 会话列表按令牌身份可见性过滤：**超管令牌必须是目标群成员**（MVP 超管视角
  口径；二期再按 viewer 权限过滤）。
- find_group 翻页上限 10 页（1000 会话）；单段消息翻页护栏默认 100 页。
