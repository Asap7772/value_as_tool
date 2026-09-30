"""The frozen 3 × 2 × 2 × 2 attempt-conditioned verifier experiment.

All variants share the mature GVR and value-tool protocols. Only the trusted
verifier receives the selected evidence pack and, for gold arms, the reference.
"""

from __future__ import annotations

from value_as_tool.harnesses._legacy import LegacyHarness
from value_as_tool.harnesses.base import ATTEMPT_CONDITIONING_MODES, HarnessSpec
from value_as_tool.schemas import Condition


def _variant(mode: str, interaction: str, feedback: str, gold: bool) -> type[LegacyHarness]:
    suffix = "gold" if gold else "no_gold"
    harness_id = f"attempt_{mode}_{interaction}_{feedback}_{suffix}"
    condition = Condition(interaction + ("_rationale_score" if feedback == "rationale" else ""))
    spec = HarnessSpec(
        harness_id=harness_id,
        display_name=(
            f"{interaction.replace('_', ' ').upper()} + {mode.replace('_', ' ')} "
            f"({feedback}, {'gold' if gold else 'no gold'})"
        ),
        access="attempt_and_reference_assisted" if gold else "attempt_assisted",
        condition=condition,
        requires_reference=gold,
        conditioning_mode=mode,
    )
    # Explicit module attribution makes source hashing and snapshot import work
    # exactly as for handwritten adapter classes.
    return type(harness_id, (LegacyHarness,), {"spec": spec, "__module__": __name__})


ATTEMPT_CONDITIONED_HARNESSES: tuple[str, ...] = tuple(
    f"{__name__}:attempt_{mode}_{interaction}_{feedback}_{'gold' if gold else 'no_gold'}"
    for mode in ATTEMPT_CONDITIONING_MODES
    for interaction in ("gvr", "value_tool")
    for feedback in ("legacy", "rationale")
    for gold in (False, True)
)

for _mode in ATTEMPT_CONDITIONING_MODES:
    for _interaction in ("gvr", "value_tool"):
        for _feedback in ("legacy", "rationale"):
            for _gold in (False, True):
                _class = _variant(_mode, _interaction, _feedback, _gold)
                globals()[_class.spec.harness_id] = _class

del _mode, _interaction, _feedback, _gold, _class

__all__ = ["ATTEMPT_CONDITIONED_HARNESSES"]
