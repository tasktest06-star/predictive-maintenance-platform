"""A hybrid digital twin for the condition-monitoring node, and its baseline.

The branch exists to answer one question adversarially: *does a digital twin
detect a developing fault earlier than the feature-space one-class detector this
platform already ships?* Everything in this package is arranged so that the
answer can come out "no" without any of it having to be rewritten.

Layout
------
``physics``    A forward model of the observable quantities -- lines at shaft
               orders, a structural floor, the datasheet sensor chain -- with
               five identified coefficients. Deliberately *not*
               :mod:`pdmcore.sim`; see ``twin/research/00_design.md`` for why
               using the simulator as the twin would make the experiment
               circular.
``residual``   A kernel-ridge correction to the physics, with a capacity knob
               and a measured effective degrees of freedom, because the central
               trap on this branch is a residual model that absorbs the fault.
``filters``    An unscented Kalman filter and a bootstrap particle filter over
               the twin's state, with divergence instrumentation.
``detect``     A CUSUM/GLR test on the whitened innovation, with the alarm
               discipline the platform uses elsewhere.
``virtual``    Load torque estimated from quantities a node already transmits,
               scored against the held-out truth and against trivial regression.
``data``       The datasets, cached, and the mapping from the 63-feature
               contract to the twin's observation vector.

Nothing here invents a measurement. ``pdmcore`` does the DSP, the features, the
baseline detector, the metrics and the alarm state machine; this package adds
the twin and the comparison.
"""

from __future__ import annotations

#: Bumped when anything changes the twin's observation vector or physics
#: parameterisation, for the same reason ``FEATURE_SPEC_VERSION`` exists: a
#: residual model fitted against one observation layout must never be fed
#: another.
TWIN_SPEC_VERSION = 1

__all__ = ["TWIN_SPEC_VERSION"]
