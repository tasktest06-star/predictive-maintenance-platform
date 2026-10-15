"""Machine signal simulation.

Lets the entire platform -- edge feature extraction, cloud diagnosis, alarm
logic, prognostics, dashboards and the per-kit prototypes -- be exercised and
regression-tested with no hardware attached. See :mod:`pdmcore.sim.machine` for
the scope and the limits of the model.
"""

from __future__ import annotations

from .electrical import (
    CouplingModel,
    CurrentSensorSpec,
    ElectricalResult,
    SupplySpec,
    mech_transfer,
    simulate_electrical,
)
from .machine import (
    ELECTRICAL_FAULT_MODES,
    FAULT_MODES,
    MACHINE_LIBRARY,
    MachineSpec,
    Resonance,
    SensorSpec,
    SimResult,
    degradation_series,
    fleet_dataset,
    simulate,
)

__all__ = [
    "MachineSpec",
    "SensorSpec",
    "Resonance",
    "SimResult",
    "FAULT_MODES",
    "ELECTRICAL_FAULT_MODES",
    "MACHINE_LIBRARY",
    "simulate",
    "degradation_series",
    "fleet_dataset",
    "SupplySpec",
    "CurrentSensorSpec",
    "CouplingModel",
    "ElectricalResult",
    "simulate_electrical",
    "mech_transfer",
]
