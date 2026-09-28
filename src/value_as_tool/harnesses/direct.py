from value_as_tool.harnesses._legacy import LegacyHarness
from value_as_tool.harnesses.base import HarnessSpec
from value_as_tool.schemas import Condition


class AgentHarness(LegacyHarness):
    spec = HarnessSpec("direct", "Direct", condition=Condition.DIRECT)
