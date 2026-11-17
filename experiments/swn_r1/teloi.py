"""SWN-9-inspired Telos/Teloi: typed feasibility and Pareto post-run audit.

Ground truth is NEVER available to generation/selection. Objectives compare
only observed runs AFTER decoding. A pairwise comparison is not causal proof.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Objective:
    name: str
    direction: str  # min or max

    def __post_init__(self):
        if not self.name or self.direction not in ("min", "max"):
            raise ValueError("objective requires a name and min/max direction")


@dataclass(frozen=True)
class Teloi:
    objectives: tuple[Objective, ...]

    def __post_init__(self):
        if not self.objectives or len({o.name for o in self.objectives}) != len(self.objectives):
            raise ValueError("Teloi needs unique objectives")

    def compare(self, proposed: dict[str, float], reference: dict[str, float]) -> str:
        for obj in self.objectives:
            for row in (proposed, reference):
                if obj.name not in row or not isfinite(float(row[obj.name])):
                    raise ValueError("missing/nonfinite metric: " + obj.name)
        better = worse = False
        for obj in self.objectives:
            a, b = float(proposed[obj.name]), float(reference[obj.name])
            better_here = a > b if obj.direction == "max" else a < b
            worse_here = a < b if obj.direction == "max" else a > b
            better |= better_here
            worse |= worse_here
        if better and not worse:
            return "proposed_pareto_dominates"
        if worse and not better:
            return "reference_pareto_dominates"
        if not better and not worse:
            return "equal"
        return "incomparable"


EVALUATION_TELOI = Teloi((
    Objective("accuracy", "max"),
    Objective("mean_steps", "min"),
    Objective("mean_seconds", "min"),
    Objective("fallbacks_per_example", "min"),
))
