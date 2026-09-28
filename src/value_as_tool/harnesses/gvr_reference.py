from value_as_tool.harnesses._legacy import LegacyHarness
from value_as_tool.harnesses.base import HarnessSpec
from value_as_tool.schemas import Condition


class AgentHarness(LegacyHarness):
    spec = HarnessSpec(
        "gvr_reference",
        "GVR + reference",
        access="reference_assisted",
        condition=Condition.GVR_REFERENCE,
        requires_reference=True,
    )
