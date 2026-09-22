"""Concurrency-safe generated-token accounting primitives.

Requests reserve their advertised maximum before being dispatched.  This is
important for the subagent condition: several simultaneous requests may each
be individually legal while their combined maximum would exceed the shared
trajectory budget.  Settlement bills exact provider-reported completion
tokens and releases the unused reservation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .schemas import TokenUsage


class BudgetError(RuntimeError):
    pass


class BudgetExhausted(BudgetError):
    pass


class BudgetAccountingError(BudgetError):
    pass


@dataclass(frozen=True)
class BudgetPermit:
    id: int
    label: str
    max_tokens: int


@dataclass(frozen=True)
class BudgetCharge:
    permit_id: int
    label: str
    requested_max_tokens: int
    usage: TokenUsage


@dataclass(frozen=True)
class UnknownUsage:
    permit_id: int
    label: str
    upper_bound_tokens: int
    reason: str


class TokenBudget:
    """One generated-token ledger shared by every role in a trajectory."""

    def __init__(self, max_generated_tokens: int) -> None:
        if max_generated_tokens <= 0:
            raise ValueError("max_generated_tokens must be positive")
        self.max_generated_tokens = max_generated_tokens
        self._next_id = 0
        self._spent_generated = 0
        self._spent_prompt = 0
        self._spent_total = 0
        self._reasoning_tokens = 0
        self._cached_prompt_tokens = 0
        self._reserved: dict[int, BudgetPermit] = {}
        self._charges: list[BudgetCharge] = []
        self._unknown: list[UnknownUsage] = []

    @property
    def spent_generated_tokens(self) -> int:
        return self._spent_generated

    @property
    def spent_prompt_tokens(self) -> int:
        return self._spent_prompt

    @property
    def reserved_generated_tokens(self) -> int:
        return sum(permit.max_tokens for permit in self._reserved.values())

    @property
    def remaining_generated_tokens(self) -> int:
        return max(
            0,
            self.max_generated_tokens - self._spent_generated - self.reserved_generated_tokens,
        )

    @property
    def has_unknown_usage(self) -> bool:
        return bool(self._unknown)

    def allowance(self, cap: int, *, keep: int = 0) -> int:
        if cap < 0 or keep < 0:
            raise ValueError("cap and keep must be non-negative")
        return max(0, min(cap, self.remaining_generated_tokens - keep))

    def reserve(self, cap: int, *, keep: int = 0, label: str) -> BudgetPermit:
        maximum = self.allowance(cap, keep=keep)
        if maximum <= 0:
            raise BudgetExhausted(
                f"no generated tokens available for {label!r} while keeping {keep}"
            )
        permit = BudgetPermit(id=self._next_id, label=label, max_tokens=maximum)
        self._next_id += 1
        self._reserved[permit.id] = permit
        return permit

    def settle(self, permit: BudgetPermit, usage: TokenUsage) -> BudgetCharge:
        active = self._pop_active(permit)
        charge = BudgetCharge(
            permit_id=active.id,
            label=active.label,
            requested_max_tokens=active.max_tokens,
            usage=usage,
        )
        # Account before raising so a provider protocol violation never makes
        # measured usage disappear from artifacts.
        self._spent_generated += usage.completion_tokens
        self._spent_prompt += usage.prompt_tokens
        self._spent_total += usage.total_tokens
        self._reasoning_tokens += usage.reasoning_tokens or 0
        self._cached_prompt_tokens += usage.cached_prompt_tokens or 0
        self._charges.append(charge)
        if usage.completion_tokens > active.max_tokens:
            raise BudgetAccountingError(
                f"{active.label!r} reported {usage.completion_tokens} completion "
                f"tokens after a max_tokens={active.max_tokens} request"
            )
        if self._spent_generated > self.max_generated_tokens:
            raise BudgetAccountingError("provider usage exceeded the trajectory budget")
        return charge

    def invalidate(self, permit: BudgetPermit, reason: str) -> UnknownUsage:
        active = self._pop_active(permit)
        unknown = UnknownUsage(
            permit_id=active.id,
            label=active.label,
            upper_bound_tokens=active.max_tokens,
            reason=reason,
        )
        self._unknown.append(unknown)
        return unknown

    def cancel(self, permit: BudgetPermit) -> None:
        """Release a reservation known not to have reached a provider."""

        self._pop_active(permit)

    def total_usage(self) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self._spent_prompt,
            completion_tokens=self._spent_generated,
            total_tokens=self._spent_total,
            reasoning_tokens=self._reasoning_tokens,
            cached_prompt_tokens=self._cached_prompt_tokens,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "max_generated_tokens": self.max_generated_tokens,
            "spent_generated_tokens": self._spent_generated,
            "spent_prompt_tokens": self._spent_prompt,
            "spent_total_tokens": self._spent_total,
            "reasoning_tokens": self._reasoning_tokens,
            "cached_prompt_tokens": self._cached_prompt_tokens,
            "reserved_generated_tokens": self.reserved_generated_tokens,
            "remaining_generated_tokens": self.remaining_generated_tokens,
            "charges": [asdict(charge) for charge in self._charges],
            "unknown_usage": [asdict(item) for item in self._unknown],
            "unknown_usage_upper_bound": sum(item.upper_bound_tokens for item in self._unknown),
        }

    def _pop_active(self, permit: BudgetPermit) -> BudgetPermit:
        active = self._reserved.pop(permit.id, None)
        if active is None or active != permit:
            raise BudgetAccountingError(f"permit {permit.id} is not active")
        return active


__all__ = [
    "BudgetAccountingError",
    "BudgetCharge",
    "BudgetError",
    "BudgetExhausted",
    "BudgetPermit",
    "TokenBudget",
    "UnknownUsage",
]
