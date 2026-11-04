"""适配层：原生 HTTP（平台 V3 适配在 M3 落地 platform/ 子包）。"""
from flowagent.adapters.http import build_http_app

__all__ = ["build_http_app"]
