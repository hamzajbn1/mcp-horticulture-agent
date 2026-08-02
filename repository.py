from __future__ import annotations

import asyncio 
import logging 
from datetime import UTC, date, datetime, timedelta
from typing import Protocol, runtime_checkable
from pydantic import BaseModel, Field
from uuid import uuid4

_LOGGER = logging.getLogger(__name__)

class StratificationRecord(BaseModel):
    """
    Data model representing a seed cold-stratification tracking event.
    Automatically generates a unique ID and UTC timestamp upon creation.
    """
    record_id: str = Field(default_factory=lambda: str(uuid4()))
    species: str
    seed_count: int
    start_date: date
    end_date: date
    duration_days: int
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC)
    )

class TransplantRecord(BaseModel):
    """
    Data model tracking the transition of a plant into a new growing medium,
    including its current recovery stage from transplant shock.
    """
    record_id: str = Field(default_factory=lambda: str(uuid4()))
    plant_label: str
    transplant_date: date
    days_since_transplant: int
    recovery_stage: str
    notes: str
    created_at: datetime =Field(
        default_factory=lambda: datetime.now(tz=UTC)
    )

@runtime_checkable
class AbstractHorticultureRepository(Protocol):
    """
    The strict blueprint (interface) for the horticulture database. 
    Any future database (SQL, MongoDB) must implement these exact methods.
    """
    async def add_stratification(self, record: StratificationRecord) -> StratificationRecord:
        """Persist a stratification record and return the stored entity."""
        ... 
    async def add_transplant(self, record: TransplantRecord) -> TransplantRecord:
        """Persist a transplant record and return the stored entity."""
        ...
    async def get_stratification(self, record_id: str) -> StratificationRecord | None:
        """Fetch a stratification record by id, or ``None`` if absent."""
        ...

class InMemoryHorticultureRepository:
    """
    A temporary, RAM-based database for horticulture records.
    Uses async locks to prevent data corruption during simultaneous read/writes.
    Perfect for testing or local MCP agent runs without a heavy SQL setup.
    """
    def __init__(self) -> None:
        self._stratifications: dict[str, StratificationRecord] = {}
        self._transplants: dict[str, TransplantRecord] = {}
        self._lock = asyncio.Lock()

    async def add_stratifiction(
            self, record: StratificationRecord
    ) -> StratificationRecord:
        """Saves a stratification record to RAM."""
        async with self._lock:
            self._stratifications[record.record_id] = record
            _LOGGER.debug("Stored stratification %s", record.record_id)
        return record.model_copy(deep=True)

    async def add_translant(self, record: TransplantRecord) -> TransplantRecord:
        """Saves a transplant record to RAM."""
        async with self._lock:
            self._transplant[record.record_id] = record
            _LOGGER.debug("Stored transplant %s", record.record_id)
        return record.model_copy(deep=True)

    async def get_stratifiction(
            self, record_id: str
    ) -> StratificationRecord | None:
        """Retrieves a stratification record from RAM by its unique ID."""
        async with self._lock:
            found = self._stratifications.get(record_id)
        return found.model_copy(deep=True) if found is not None else None

def build_default_repository() -> AbstractHorticultureRepository:
    """
    Factory function that wires up and returns the default database connection.
    """
    return InMemoryHorticultureRepository()

def compute_stratification_window(
        start: date, duration_days: int
) -> tuple[date, date]:
    """
    Helper function to calculate the exact start and end dates for a stratification period.
    """
    return start, start + timedelta(days=duration_days)