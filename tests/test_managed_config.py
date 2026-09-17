import pytest

from flowagent.errors import ConfigError
from flowagent.managed_config import load_managed_spec


def _yaml(node: str) -> str:
    return f"""
workflow:
  name: managed
  start: main
  inputs: [message, history, system_prompt, skills]
nodes:
  main:
{node}
"""


def test_managed_profile_accepts_agent_node_with_bounded_iterations():
    spec = load_managed_spec(_yaml("""    type: agent
    output_key: reply
    max_tool_iterations: 8
    timeout: 90
    next: END"""))
    assert spec.nodes["main"].type == "agent"
    assert spec.nodes["main"].max_tool_iterations == 8


@pytest.mark.parametrize("node", [
    "    type: human\n    message: approve?\n    next: END",
    "    type: tool\n    tool: flowagent.tools:bump_counter\n    next: END",
    "    type: workflow\n    workflow_file: child.yaml\n    next: END",
])
def test_managed_profile_rejects_local_execution_nodes(node):
    with pytest.raises(ConfigError, match="managed_node_type_not_allowed"):
        load_managed_spec(_yaml(node))


def test_managed_profile_rejects_more_than_eight_tool_iterations():
    with pytest.raises(ConfigError):
        load_managed_spec(_yaml("""    type: agent
    output_key: reply
    max_tool_iterations: 9
    next: END"""))
