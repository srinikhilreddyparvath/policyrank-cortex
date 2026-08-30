from __future__ import annotations

from dataclasses import dataclass


class ContaminatedComponentError(RuntimeError):
    """Raised when a paper run attempts to use legacy-contaminated state."""


@dataclass(frozen=True)
class ComponentProvenance:
    name: str
    status: str
    reason: str


LEGACY_Q_TABLE = ComponentProvenance(
    "storage/slate_q_table.json",
    "legacy_contaminated_for_paper_eval",
    "historically trained from ESCI-label-derived reward",
)

LEGACY_GATE_THRESHOLDS = ComponentProvenance(
    "baseline_preservation_gate_thresholds",
    "legacy_contaminated_for_paper_eval",
    "calibration provenance cannot be proven train_fit-only",
)


def require_safe(component: ComponentProvenance, *, allow_unsafe_debug: bool = False) -> None:
    if component.status == "legacy_contaminated_for_paper_eval" and not allow_unsafe_debug:
        raise ContaminatedComponentError(
            f"Refusing {component.name}: {component.status}; {component.reason}. "
            "Use --allow-unsafe-contaminated-state for debugging only."
        )
