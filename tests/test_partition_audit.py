"""分区审计的集合校验，不启动子进程或访问外部服务。"""
import pytest

from scripts.audit_test_partitions import compare_partitions


@pytest.mark.parametrize('partitions,expected', [
    ({'live': {'a'}, 'installation': {'b'}, 'process': set(), 'db': set(), 'offline': {'c'}}, True),
    ({'live': {'a'}, 'installation': {'a', 'b'}, 'process': set(), 'db': set(), 'offline': {'c'}}, False),
    ({'live': {'a'}, 'installation': {'b'}, 'process': set(), 'db': set(), 'offline': set()}, False),
    ({'live': {'a'}, 'installation': {'b'}, 'process': set(), 'db': set(), 'offline': {'c', 'extra'}}, False),
])
def test_partition_audit_rejects_overlap_missing_and_extra(partitions, expected):
    report = compare_partitions({'a', 'b', 'c'}, partitions)
    assert report['ok'] is expected
    assert report['counts'] == {name: len(nodes) for name, nodes in partitions.items()}


def test_partition_audit_rejects_empty_collection():
    partitions = {name: set() for name in ('live', 'installation', 'process', 'db', 'offline')}
    report = compare_partitions(set(), partitions)
    assert report['ok'] is False and report['empty_collection'] is True


def test_empty_partition_accepts_pytest_exit_five(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import audit_test_partitions as audit
    monkeypatch.setattr(audit.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=5, stdout='', stderr=''))
    assert audit.collect(tmp_path, 'process and not live') == set()
