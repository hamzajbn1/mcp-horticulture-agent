from __future__ import annotations

import logging
from datetime import date

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from domain import (
    DEFAULT_ERICACEOUS_BAND,  
    Species,
    classify_recovery_stage,
    ph_correction_hint,
    recommended_stratification_days,
)

from repository import (
    AbstractHorticultureRepository,
    StratificationRecord,
    TransplantRecord,
    build_default_repository,
    compute_stratification_window,
)

_LOGGER = logging.getLogger(__name__)

class StratificationRequest(BaseModel):
    """Input schema that forces the AI to provide exact data for logging cold stratification."""
    species: Species = Field(description="Seed species: 'kiwi' or 'blueberry'.")

    seed_count: int = Field(gt=0, le=100_000, description="Number of seeds in the batch.")

    start_date: date = Field(description="ISO date (YYYY-MM-DD) chilling begins.")

class TransplantRequest(BaseModel):
    """Input schema ensuring the AI provides valid transplant recovery data."""
    plant_label: str = Field(
        min_length=1, max_length=120, description="Human-readable plant identifier."
    )
    transplant_date: date = Field(description="ISO date the plant was transplanted.")
    days_since_transplant: int = Field(
        ge=0, le=3650, description="Whole days elapsed since transplanting."
    )
    notes: str = Field(default="", max_length=1000, description="Optional observations.")

def build_server(repository: AbstractHorticultureRepository | None = None) ->FastMCP:
    """
    Constructs the FastMCP server, connects to the database repository, 
    and registers all available tools for the AI agent to use.
    """
    repo = repository if repository is not None else build_default_repository()
    mcp: FastMCP = FastMCP(name="horticulture-tracker")

    @mcp.tool
    async def log_cold_stratification(request: StratificationRequest) -> dict[str, object]:
        duration = recommended_stratification_days(request.species)
        start, end = compute_stratification_window(request.start_date, duration)

        record = StratificationRecord(
            species=request.species.value,
            seed_count=request.seed_count,
            start_date=start,
            end_date=end,
            duration_days=duration,
        )

        stored= await repo.add_stratification(record)
        _LOGGER.info(
            "Logged stratification for %s x%d (%d days)",
            stored.species,
            stored.seed_count,
            stored.duration_days,
        )
        return{
            "record_id": stored.record_id,
            "species": stored.species,
            "seed_count": stored.seed_count,
            "start_date": stored.start_date.isoformat(),
            "end_date": stored.end_date.isoformat(),
            "duration_days": stored.duration_days,
        }

    @mcp.tool
    async def validate_ericaceous_ph(soil_ph: float) -> dict[str, object]:
        """Checks if a soil pH reading is safe for acid-loving plants."""
        if not 0.0 <= soil_ph <= 14.0:
            msg = "soil_ph must be within the 0-14 range"
            raise ValueError(msg)

        band = DEFAULT_ERICACEOUS_BAND
        classification = band.classify(soil_ph)
        _LOGGER.info("Validated pH %.2f -> %s", soil_ph, classification)
        return{
            "soil_ph": soil_ph,
            "acceptable_min": band.minimum,
            "acceptable_max": band.maximum,
            "classification": classification,
            "recommendation": ph_correction_hint(classification),
        }

    @mcp.tool
    async def track_transplant_recovery(request: TransplantRequest) -> dict[str, object]:
        """Evaluates and records the transplant shock recovery stage."""
        stage = classify_recovery_stage(request.days_since_transplant)
        record = TransplantRecord(
            plant_label = request.plant_label,
            transplant_date = request.transplant_date,
            days_since_transplant = request.days_since_transplant,
            recovery_stage = stage,
            notes = request.notes,
        )
        stored = await repo.add_transplant(record)
        
        _LOGGER.info(
            "Tracked transplant '%s' -> stage=%s", stored.plant_label, stored.recovery_stage
        )
        return {
            "record_id": stored.record_id,
            "plant_label": stored.plant_label,
            "transplant_date": stored.transplant_date.isoformat(),
            "days_since_transplant": stored.days_since_transplant,
            "recovery_stage": stored.recovery_stage,
            "notes": stored.notes,
        }
    return mcp

mcp = build_server()

if __name__ == "__main__":
    mcp.run(transport="stdio")

