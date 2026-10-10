import pytest

from services.im_tools_mcp.tools import _format_message


@pytest.mark.parametrize('timestamp', ['2026-10-10T05:05:00Z', '2026-10-10T13:05:00+08:00'])
def test_group_quote_time_matches_beijing_page_time(timestamp):
    # 同一条消息无论存储采用哪个时区，都必须显示页面的北京时间。
    result = _format_message({'type': 'text', 'sender_id': 1, 'content': '演练材料',
                              'created_at': timestamp}, {1: '李工'})
    assert result == '[10-10 13:05 UTC+08:00] 李工: 演练材料'
