"""受管 execution-v1 服务的拆分模块（Stage 2：contracts/errors/logs/deps/planner）。

`flowagent/execution_service.py` 保留作为向后兼容入口，实际的模型定义、错误
分类、日志缓冲、凭据/依赖装配、flow 路由选择逐步收敛到本包，减少重复实现。
"""
