import pytest

from flowagent.group_analysis import _taxonomy


@pytest.mark.parametrize('value', [
    '通知类、风险类、待办类、其他类',
    '按通知类、风险类、待办类、其他类分类',
    '分为通知类、风险类、待办类、其他类',
    '按通知类、风险类、待办类、其他类分析。',
    '按通知类、风险类、待办类、其他类分类，汇总泵站状态和待办',
])
def test_explicit_categories_survive_instruction_prefix_and_suffix(value):
    # 模型传入分类指令或纯名称列表时，应采用同一组明确分类。
    assert _taxonomy(value) == ('通知类', '风险类', '待办类', '其他类', '未归类')


@pytest.mark.parametrize('value', ['自定义', '默认分类', '风险类'])
def test_incomplete_categories_still_require_clarification(value):
    with pytest.raises(ValueError, match='taxonomy_needs_explicit_categories'):
        _taxonomy(value)


def test_existing_message_nature_prefix_remains_supported():
    # 修复指令后缀时，也保留既有的“按消息性质分”写法。
    assert _taxonomy('按消息性质分技术类/管理类') == ('技术类', '管理类', '未归类')
