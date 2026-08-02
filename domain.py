from __future__ import annotations

from enum import StrEnum
from typing import Final

from pydantic import BaseModel, Field

# Creates a strict drop-down menu of allowed plants.
class Species(StrEnum):
    KIWI = "kiwi"
    BLUEBERRY= "blueberry"

# Constant mapping of plant species to their required cold stratification days
STRATIFICATION_DURATIONS_DAYS: Final[dict[Species, int]] = {
    Species.KIWI: 90,
    Species.BLUEBERRY: 70,
}

class EricaceousPHBand(BaseModel):
    """Acceptable soil pH band for ericaceous (acid-loving) plants."""
    minimum: float = Field(default=4.5, description="Lower acceptable pH bound.")
    maximum: float = Field(default=5.5, description="Upper acceptable pH bound.")

    def classify(self, ph: float) -> str:
        if ph < self.minimum:
            return "too_acidic"
        if ph > self.maximum:
            return "too_alkaline"
        return "optimal"

# Standard global constant for ericaceous soil bounds
DEFAULT_ERICACEOUS_BAND: Final[EricaceousPHBand] = EricaceousPHBand()

# Constant mapping of day thresholds to transplant recovery stages
_RECOVERY_STAGES: Final[tuple[tuple[int, str], ...]] = (
    (0, "acute_shock"),
    (4, "early_recovery"),
    (11, "stabilising"),
    (22, "established"),
)

def recommended_stratification_days(species: Species) -> int:
    return STRATIFICATION_DURATIONS_DAYS[species]

# Determine the current recovery stage of a transplanted plant based on elapsed days.
def classify_recovery_stage(days_since_transplant: int) -> str:
    if days_since_transplant < 0:
        msg = "days_since_transplant must be non-negative"
        raise ValueError(msg)

    stage = _RECOVERY_STAGES[0][1]
    for lower_bound, label in _RECOVERY_STAGES:
        if days_since_transplant >= lower_bound:
            stage= label
        else: 
            break 
    return stage

# Provide specific soil amendment advice based on the pH classification.
def ph_correction_hint(classification: str) -> str:
    hint = {
        "too_acidic": "Raise pH with dolomitic lime; retest after 2-3 weeks.",
        "too_alkaline": "Lower pH with elemental sulfur or an ericaceous mix.",
        "optimal": "No amendment required; maintain current regime.",
    }
    return hint[classification]