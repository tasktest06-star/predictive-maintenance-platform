"""The decision layer: measurement -> action -> money.

The chain the rest of ``pdmcore`` builds up to:

    features -> health index -> alarm state -> RUL
             -> scenario-conditional RUL (:mod:`pdmcore.health.scenario_rul`)
             -> per-asset decision (:mod:`pdmcore.decide.policy`)
             -> fleet schedule (:mod:`pdmcore.decide.schedule`)
             -> work-order recommendation

and, running alongside it, the question that decides whether any of the chain
was worth building:

    :mod:`pdmcore.decide.voi`    how much is a better estimate worth?
    :mod:`pdmcore.decide.costs`  what does being wrong cost, and who said so?
    :mod:`pdmcore.decide.counts` every rate carries the count behind it.

Each stage can abstain, and abstention propagates: a RUL that refuses produces
a decision that refuses, and an asset that refuses is left out of the schedule
with a reason rather than scheduled on an invented posterior.
"""

from __future__ import annotations

from .costs import (
    DEFAULT_PROVENANCE,
    SITE_PROFILES,
    CostModelNotConfigured,
    MaintenanceCostModel,
    ProgrammeCost,
)
from .counts import CountedRate, SmallCountError, poisson_count_band, poisson_rate_ci, require_counts
from .policy import (
    RUN_TO_FAILURE,
    ActionEvaluation,
    Decision,
    Opportunity,
    decide,
    evaluate_opportunities,
    programme_balance,
    shutdown_calendar,
)
from .schedule import AssetJob, FleetScheduler, ScheduleResult, ShutdownWindow
from .voi import (
    PrecisionPoint,
    VoIStudy,
    break_even_precision,
    decision_relevant_precision,
    interval_width_days,
    log_sd_for_width,
    precision_step_value,
    precision_table,
    realised_cost,
    voi_curve,
)

__all__ = [
    "MaintenanceCostModel", "ProgrammeCost", "CostModelNotConfigured",
    "DEFAULT_PROVENANCE", "SITE_PROFILES",
    "CountedRate", "SmallCountError", "poisson_rate_ci", "poisson_count_band",
    "require_counts",
    "Opportunity", "RUN_TO_FAILURE", "ActionEvaluation", "Decision",
    "shutdown_calendar", "evaluate_opportunities", "decide", "programme_balance",
    "AssetJob", "ShutdownWindow", "ScheduleResult", "FleetScheduler",
    "VoIStudy", "PrecisionPoint", "voi_curve", "decision_relevant_precision",
    "break_even_precision", "precision_table", "precision_step_value",
    "interval_width_days", "log_sd_for_width", "realised_cost",
]
