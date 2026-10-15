"""Health index, alarming, diagnosis and prognostics.

The chain is: feature vector -> health index (:mod:`pdmcore.health.index`) ->
alarm state -> rule-based diagnosis (:mod:`pdmcore.health.rules`) -> RUL
(:mod:`pdmcore.health.rul`) -> work order. Each stage is separately testable and
each stage can abstain, which is what keeps a confident wrong answer from
reaching a maintenance planner.
"""

from __future__ import annotations

from .index import (
    AlarmConfig,
    AlarmEvent,
    AlarmState,
    AlarmStateMachine,
    EwmaTrend,
    HealthIndex,
    OperatingBaseline,
)
from .rul import ParticleFilterRUL, RULEstimate, fit_exponential_trend
from .rules import RULES, Diagnosis, RuleContext, baseline_from_vectors, diagnose, register_rule
from .scenario_rul import (
    BALL_BEARING_EXPONENT,
    ROLLER_BEARING_EXPONENT,
    LoadScenario,
    ScenarioRUL,
    ScenarioRULEstimate,
)

__all__ = [
    "AlarmState", "AlarmConfig", "AlarmEvent", "AlarmStateMachine",
    "EwmaTrend", "OperatingBaseline", "HealthIndex",
    "Diagnosis", "RuleContext", "RULES", "diagnose", "register_rule", "baseline_from_vectors",
    "RULEstimate", "ParticleFilterRUL", "fit_exponential_trend",
    "LoadScenario", "ScenarioRUL", "ScenarioRULEstimate",
    "BALL_BEARING_EXPONENT", "ROLLER_BEARING_EXPONENT",
]
