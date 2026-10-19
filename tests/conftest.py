"""测试基座：仅提供项目根路径；旧引擎/PG 全局 fixture 已随 M4 退役。"""
from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
