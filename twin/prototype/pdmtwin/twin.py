"""The hybrid twin, assembled: physics + learned residual + state estimator.

This is the object the head-to-head runs. It is built so that every structural
choice the branch is testing is a constructor argument -- number of free states,
residual model, residual inputs, whether the residual is refitted online -- and
so that the three windows it needs (fit / calibrate / verify) cannot be
confused with each other, because each is a separate method call that takes its
own data.

The pipeline, once, in order:

1. ``identify``  -- probe the feature extractor, fit the five physics
   coefficients to the **fit** window's median frame.
2. ``fit_residual`` -- point-estimate the state per fit-window frame, take the
   log residual, fit the correction on the **fit** window only.
3. ``calibrate`` -- on the **calibration** window: the innovation covariance R,
   the standardisation of the per-frame deviation, and the CUSUM threshold that
   achieves a target ARL0.
4. ``verify`` -- on the **verification** window: the realised alarm count
   against a Poisson band. This is the step that can fail, and it is the whole
   reason the windows are separate.

Steps 3 and 4 are different data on purpose. ``docs/08_branch_catalogue.md``
records two defects of the form "calibrated and verified on the same window, so
the check could never fail"; this is the same shape of mistake one level up, and
the structure here makes it awkward to make.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pdmcore.features import FeatureExtractor
from pdmcore.sim import MachineSpec

from .data import Dataset, ObsSpec
from .detect import (
    CusumTest,
    energy_deviation,
    measure_arl0,
    poisson_allowance,
    robust_centre_scale,
    threshold_for_arl0,
)
from .filters import FilterTrace, ProcessModel, UnscentedTwinFilter
from .physics import ObservationModel, PhysicsCore, identify
from .residual import NullResidual, ResidualInputs, design_matrix

__all__ = ["HybridTwin", "TwinConfig"]


@dataclass
class TwinConfig:
    """Everything that changes the twin's numbers, in one object."""

    n_state: int = 2
    residual_kind: str = "kernel_ridge"
    residual_inputs: ResidualInputs = ResidualInputs.EXOGENOUS
    residual_alpha: float = 1.0
    residual_hidden: Tuple[int, ...] = (16,)
    #: Refit the residual on a sliding window of recent frames, including
    #: possibly-faulty ones. This is the "the twin tracks ageing" feature, and
    #: it is absorption mechanism 1. Off by default; measured when on.
    online_refit: int = 0
    #: ``"ukf"``, ``"point"`` (per-frame Gauss-Newton, no filter) or ``"pf"``
    #: (bootstrap particle filter). The ``"point"`` arm is not a strawman: a
    #: recursive filter can only beat a per-frame estimate by exploiting
    #: temporal correlation in the state, and whether the state *is* correlated
    #: is a property of the data, not of the filter. See ``process_from_data``.
    filter_mode: str = "ukf"
    #: Identify the process model (AR(1) coefficient and innovation std per
    #: state) from the fit window instead of assuming it. This is what turned
    #: up the branch's most uncomfortable measurement; the fallback constants
    #: below are used only when ``process_from_data=False``.
    process_from_data: bool = True
    process_sigma: Tuple[float, ...] = (0.10, 0.04, 0.02)
    process_revert: Tuple[float, ...] = (0.25, 0.25, 0.0)
    #: Rescale R so the filter's own innovations are as large as its covariance
    #: says. Without it the filter is over-confident by a factor of 3-4 here,
    #: because R estimated from point-estimate residuals is biased low by the
    #: states that were fitted out of it.
    adapt_R: bool = True
    cusum_drift: float = 0.5
    consecutive_required: int = 2
    use_full_R: bool = True

    def as_dict(self) -> Dict:
        return {
            "n_state": self.n_state,
            "residual_kind": self.residual_kind,
            "residual_inputs": self.residual_inputs.value,
            "residual_alpha": self.residual_alpha,
            "residual_hidden": list(self.residual_hidden),
            "online_refit": self.online_refit,
            "filter_mode": self.filter_mode,
            "process_from_data": self.process_from_data,
            "process_sigma": list(self.process_sigma),
            "process_revert": list(self.process_revert),
            "adapt_R": self.adapt_R,
            "cusum_drift": self.cusum_drift,
            "consecutive_required": self.consecutive_required,
            "use_full_R": self.use_full_R,
        }


class HybridTwin:
    """Physics core, learned residual, UKF, and a CUSUM on the innovation."""

    def __init__(
        self,
        spec: MachineSpec,
        extractor: FeatureExtractor,
        obs: ObsSpec,
        config: Optional[TwinConfig] = None,
    ):
        self.spec = spec
        self.extractor = extractor
        self.obs = obs
        self.config = config or TwinConfig()
        self.core: Optional[PhysicsCore] = None
        self.residual = NullResidual(n_channels=len(obs), inputs=self.config.residual_inputs)
        self.sd0: np.ndarray = np.ones(len(obs))
        self.R: np.ndarray = np.eye(len(obs))
        self.q_centre: float = 0.0
        self.q_scale: float = 1.0
        self.process: ProcessModel = ProcessModel(
            n_state=self.config.n_state,
            revert=tuple(self.config.process_revert),
            sigma=tuple(self.config.process_sigma),
        )
        self.test: CusumTest = CusumTest(drift=self.config.cusum_drift,
                                         consecutive_required=self.config.consecutive_required)
        self.reports: Dict[str, Dict] = {}

    # ------------------------------------------------------------------ 1 --
    def identify(self, ds_fit: Dataset, *, model: Optional[ObservationModel] = None) -> "HybridTwin":
        Y = ds_fit.obs(self.obs)
        self.core = identify(
            self.spec, self.extractor, self.obs, Y, ds_fit.shaft_hz,
            n_state=self.config.n_state, model=model,
        )
        resid = np.log(np.maximum(Y, 1e-12)) - self.core.predict_log(
            self.core.nominal_state(ds_fit.n), ds_fit.shaft_hz
        )
        self.sd0 = np.maximum(np.std(resid, axis=0), 1e-6)
        self.reports["physics"] = dict(self.core.fit_report)
        self.reports["physics"]["healthy_log_sd_at_nominal"] = [float(v) for v in self.sd0]
        return self

    # -- state point estimate (used for the residual fit and virtual sensing)
    def point_estimate(self, Y: np.ndarray, shaft_hz: np.ndarray, *, n_iter: int = 4) -> np.ndarray:
        """Per-frame maximum-likelihood state by Gauss-Newton in log space.

        Four iterations with a numerical Jacobian. Used where a recursive filter
        is the wrong tool: fitting the residual model needs a state estimate
        that does not depend on the residual model, and a virtual sensor reading
        for a single frame has no history to filter.
        """
        assert self.core is not None
        Z = np.log(np.maximum(np.atleast_2d(Y), 1e-12))
        n, c = Z.shape
        ns = self.core.n_state
        x = np.zeros((n, ns))
        w = 1.0 / self.sd0
        eps = 1e-4
        for _ in range(n_iter):
            h0 = self.core.predict_log(x, shaft_hz)
            J = np.zeros((n, c, ns))
            for j in range(ns):
                xp = x.copy()
                xp[:, j] += eps
                J[:, :, j] = (self.core.predict_log(xp, shaft_hz) - h0) / eps
            r = (Z - h0) * w
            Jw = J * w[None, :, None]
            A = np.einsum("ncj,nck->njk", Jw, Jw) + 1e-8 * np.eye(ns)[None]
            b = np.einsum("ncj,nc->nj", Jw, r)
            x = np.clip(x + np.linalg.solve(A, b[:, :, None])[:, :, 0], -3.0, 3.0)
        return x

    # -- process identification --------------------------------------------
    def identify_process(self, ds_fit: Dataset) -> "HybridTwin":
        """Fit an AR(1) process to the point-estimated state, per trajectory.

        The result decides whether the filter can help at all. A Kalman filter
        beats a per-frame estimate only by borrowing information from
        neighbouring frames, and how much there is to borrow is the lag-1
        autocorrelation of the state. If it is zero the filter's prediction step
        carries no information and the posterior reduces to the per-frame
        estimate shrunk toward the prior -- which is *worse* than the per-frame
        estimate for detection, because the shrinkage pulls the state toward
        nominal and therefore pushes the fault into the innovation... and also
        pulls the prediction away from the measurement on healthy frames, which
        inflates the healthy innovation spread by the same mechanism. The net
        effect is measured, not assumed.

        ``rho`` near 0 is not a modelling failure here, it is a property of
        ``pdmcore.sim``: ``load_factor`` and ``resonance_gain`` are drawn
        independently in every call to :func:`simulate`, so the nuisance state
        of a simulated asset is white noise. A real pump holds its duty for
        hours. ``run_autocorrelation.py`` builds datasets with a controlled
        AR(1) nuisance to show where the filter starts to pay.
        """
        assert self.core is not None
        rhos: List[List[float]] = [[] for _ in range(self.core.n_state)]
        sds: List[List[float]] = [[] for _ in range(self.core.n_state)]
        for t in np.unique(ds_fit.traj):
            idx = np.flatnonzero(ds_fit.traj == t)
            idx = idx[np.argsort(ds_fit.t_days[idx], kind="mergesort")]
            sub = ds_fit.take(idx)
            x = self.point_estimate(sub.obs(self.obs), sub.shaft_hz)
            for j in range(self.core.n_state):
                v = x[:, j] - np.mean(x[:, j])
                if v.size < 8 or np.std(v) <= 0:
                    continue
                rho = float(np.sum(v[1:] * v[:-1]) / max(1e-12, np.sum(v[:-1] ** 2)))
                rho = float(np.clip(rho, -0.95, 0.95))
                rhos[j].append(rho)
                sds[j].append(float(np.std(v) * np.sqrt(max(1e-6, 1.0 - rho**2))))
        rho_hat = [float(np.median(r)) if r else 0.0 for r in rhos]
        sd_hat = [float(np.median(s)) if s else 0.05 for s in sds]
        if self.config.process_from_data:
            self.process = ProcessModel(
                n_state=self.core.n_state,
                revert=tuple(1.0 - np.asarray(rho_hat)),
                sigma=tuple(np.maximum(sd_hat, 1e-3)),
            )
        self.reports["process"] = {
            "rho_lag1_per_state": rho_hat,
            "innovation_sd_per_state": sd_hat,
            "state_sd_per_state": [float(np.median(s) / np.sqrt(max(1e-6, 1 - r**2)))
                                   for s, r in zip(sd_hat, rho_hat)],
            "from_data": bool(self.config.process_from_data),
            "revert_used": [float(v) for v in self.process.revert[: self.core.n_state]],
            "sigma_used": [float(v) for v in self.process.sigma[: self.core.n_state]],
        }
        return self

    # ------------------------------------------------------------------ 2 --
    def fit_residual(self, ds_fit: Dataset) -> "HybridTwin":
        """Fit the correction on healthy fit-window frames only."""
        assert self.core is not None
        from .residual import build_residual

        Y = ds_fit.obs(self.obs)
        Z = np.log(np.maximum(Y, 1e-12))
        x_hat = self.point_estimate(Y, ds_fit.shaft_hz)
        Rlog = Z - self.core.predict_log(x_hat, ds_fit.shaft_hz)
        kwargs: Dict = {"inputs": self.config.residual_inputs}
        if self.config.residual_kind in ("kernel_ridge", "krr"):
            kwargs["alpha"] = self.config.residual_alpha
        elif self.config.residual_kind == "mlp":
            kwargs["hidden"] = tuple(self.config.residual_hidden)
            kwargs["alpha"] = self.config.residual_alpha
        self.residual = build_residual(self.config.residual_kind, len(self.obs), **kwargs)
        if not isinstance(self.residual, NullResidual):
            U = design_matrix(self.config.residual_inputs, shaft_hz=ds_fit.shaft_hz,
                              state=x_hat, z=Z, f_nom=self.spec.shaft_hz)
            self.residual.fit(U, Rlog)
        self.reports["residual"] = dict(self.residual.report())
        self.reports["residual"]["healthy_log_sd_before"] = [float(v) for v in np.std(Rlog, axis=0)]
        if not isinstance(self.residual, NullResidual):
            U = design_matrix(self.config.residual_inputs, shaft_hz=ds_fit.shaft_hz,
                              state=x_hat, z=Z, f_nom=self.spec.shaft_hz)
            after = Rlog - self.residual.predict(U)
            self.reports["residual"]["healthy_log_sd_after"] = [float(v) for v in np.std(after, axis=0)]
        return self

    # -- measurement function fed to the filter ----------------------------
    def _h(self, pts: np.ndarray, aux: Dict) -> np.ndarray:
        assert self.core is not None
        shaft = np.full(pts.shape[0], aux["shaft_hz"])
        base = self.core.predict_log(pts, shaft)
        if isinstance(self.residual, NullResidual):
            return base
        z = np.repeat(np.atleast_2d(aux["z"]), pts.shape[0], axis=0)
        U = design_matrix(self.config.residual_inputs, shaft_hz=shaft, state=pts, z=z,
                          f_nom=self.spec.shaft_hz)
        return base + self.residual.predict(U)

    def run_trajectory(self, Y: np.ndarray, shaft_hz: np.ndarray) -> Dict[str, np.ndarray]:
        """Run the estimator over one trajectory in time order.

        Returns the standardised deviation stream ``q`` that the sequential test
        consumes, plus the state track and the filter diagnostics. All three
        filter modes return the same keys so the harness cannot tell them apart.
        """
        assert self.core is not None
        Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
        Z = np.log(np.maximum(Y, 1e-12))
        mode = self.config.filter_mode
        if mode == "point":
            x = self.point_estimate(Y, shaft_hz)
            resid = Z - self._h_rows(x, shaft_hz, Z)
            L = np.linalg.cholesky(self.R)
            wht = np.linalg.solve(L, resid.T).T
            q_raw = energy_deviation(wht)
            nis = np.sum(wht**2, axis=1)
            return {
                "q": (q_raw - self.q_centre) / self.q_scale, "q_raw": q_raw,
                "state": x, "nis": nis, "innovation": resid,
                "diagnostics": {"nis_over_dof_mean": float(np.mean(nis) / Z.shape[1]),
                                "nis_over_dof_median": float(np.median(nis) / Z.shape[1]),
                                "n_frames": int(Z.shape[0]), "n_chol_repairs": 0,
                                "n_sigma_clipped": 0, "max_innovation_cov_cond": 1.0,
                                "mode": "point"},
            }
        aux = [{"shaft_hz": float(shaft_hz[k]), "z": Z[k]} for k in range(Z.shape[0])]
        if mode == "pf":
            from .filters import BootstrapTwinFilter
            pf = BootstrapTwinFilter(self._h, self.R, self.process)
            out = pf.run(Z, aux)
            resid = Z - self._h_rows(out["x"], shaft_hz, Z)
            L = np.linalg.cholesky(self.R)
            wht = np.linalg.solve(L, resid.T).T
            q_raw = energy_deviation(wht)
            nis = np.sum(wht**2, axis=1)
            return {
                "q": (q_raw - self.q_centre) / self.q_scale, "q_raw": q_raw,
                "state": out["x"], "nis": nis, "innovation": resid,
                "diagnostics": {"nis_over_dof_mean": float(np.mean(nis) / Z.shape[1]),
                                "nis_over_dof_median": float(np.median(nis) / Z.shape[1]),
                                "n_frames": int(Z.shape[0]), "n_resamples": out["n_resamples"],
                                "n_chol_repairs": 0, "n_sigma_clipped": 0,
                                "max_innovation_cov_cond": 1.0, "mode": "pf"},
            }
        if mode != "ukf":
            raise ValueError(f"unknown filter_mode {mode!r}")
        ukf = UnscentedTwinFilter(self._h, self.R, self.process)
        trace = ukf.run(Z, aux)
        q_raw = energy_deviation(trace.whitened)
        return {
            "q": (q_raw - self.q_centre) / self.q_scale,
            "q_raw": q_raw,
            "state": trace.x,
            "nis": trace.nis,
            "innovation": trace.innovation,
            "diagnostics": ukf.diagnostics(trace),
        }

    def run_trajectory_online(
        self, Y: np.ndarray, shaft_hz: np.ndarray, *, window: int, stride: int = 10
    ) -> Dict[str, np.ndarray]:
        """Absorption mechanism 1: refit the residual as the trajectory runs.

        "The twin keeps learning, so it tracks ageing" is a selling point and a
        self-inflicted wound. Every ``stride`` frames the residual model is
        refitted on the trailing ``window`` frames of *this* trajectory, which
        on a degrading asset means training on the fault. Anything slower than
        the window is learned as normal and vanishes from the innovation.

        The filter state is carried across blocks, so the only difference from
        :meth:`run_trajectory` is which data the residual saw. That is the point
        of measuring it this way: the mechanism is isolated.
        """
        assert self.core is not None
        from .residual import build_residual

        Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
        Z = np.log(np.maximum(Y, 1e-12))
        n = Z.shape[0]
        process = self.process
        ukf = UnscentedTwinFilter(self._h, self.R, process)
        saved = self.residual
        q_raw = np.zeros(n)
        wht = np.zeros_like(Z)
        xs = np.zeros((n, self.core.n_state))
        nis = np.zeros(n)
        n_refits = 0
        try:
            for k in range(n):
                if window > 0 and k >= window and k % stride == 0:
                    lo = k - window
                    Yw, fw = Y[lo:k], shaft_hz[lo:k]
                    xw = self.point_estimate(Yw, fw)
                    Rw = np.log(np.maximum(Yw, 1e-12)) - self.core.predict_log(xw, fw)
                    kwargs: Dict = {"inputs": self.config.residual_inputs}
                    if self.config.residual_kind in ("kernel_ridge", "krr"):
                        kwargs["alpha"] = self.config.residual_alpha
                    new = build_residual(self.config.residual_kind, len(self.obs), **kwargs)
                    if not isinstance(new, NullResidual):
                        U = design_matrix(self.config.residual_inputs, shaft_hz=fw, state=xw,
                                          z=np.log(np.maximum(Yw, 1e-12)), f_nom=self.spec.shaft_hz)
                        new.fit(U, Rw)
                    self.residual = new
                    n_refits += 1
                inn, w, q = ukf.step(Z[k], {"shaft_hz": float(shaft_hz[k]), "z": Z[k]})
                wht[k] = w
                nis[k] = q
                xs[k] = ukf.x
            q_raw = energy_deviation(wht)
        finally:
            self.residual = saved
        return {
            "q": (q_raw - self.q_centre) / self.q_scale, "q_raw": q_raw,
            "state": xs, "nis": nis, "innovation": wht,
            "diagnostics": {**ukf.diagnostics(FilterTrace(
                x=xs, P=np.zeros((n, 1, 1)), innovation=wht, whitened=wht, nis=nis)),
                "n_refits": n_refits, "mode": "ukf_online"},
        }

    def _h_rows(self, x: np.ndarray, shaft_hz: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Row-wise measurement prediction (one state per frame)."""
        assert self.core is not None
        base = self.core.predict_log(x, shaft_hz)
        if isinstance(self.residual, NullResidual):
            return base
        U = design_matrix(self.config.residual_inputs, shaft_hz=shaft_hz, state=x, z=Z,
                          f_nom=self.spec.shaft_hz)
        return base + self.residual.predict(U)

    def _traj_index(self, ds: Dataset) -> List[np.ndarray]:
        """Row indices per trajectory, each in time order."""
        out = []
        for t in np.unique(ds.traj):
            idx = np.flatnonzero(ds.traj == t)
            out.append(idx[np.argsort(ds.t_days[idx], kind="mergesort")])
        return out

    def _by_trajectory(self, ds: Dataset) -> List[np.ndarray]:
        out = []
        for idx in self._traj_index(ds):
            sub = ds.take(idx)
            out.append(self.run_trajectory(sub.obs(self.obs), sub.shaft_hz)["q"])
        return out

    # ------------------------------------------------------------------ 3 --
    def calibrate(
        self, ds_fit: Dataset, ds_cal: Dataset, *, target_arl0_frames: float = 720.0
    ) -> "HybridTwin":
        """Innovation covariance, deviation standardisation, CUSUM threshold.

        ``R`` is estimated from the **calibration** window, not the fit window.
        Estimating it from the data the residual model was fitted on is
        absorption mechanism 3 in its purest form: an overfitted residual makes
        the fit-window residuals small, ``R`` small, the innovations apparently
        significant, and the threshold is then set on a spread that does not
        describe any frame the twin has not already seen. The optimistic variant
        is measured in ``run_capacity.py`` and costs a factor of several in
        realised false-alarm rate.

        ``target_arl0_frames = 720`` is 180 days at a 6 h cadence: one nuisance
        alarm per asset per six months, which is the budget a maintenance team
        will actually tolerate.
        """
        assert self.core is not None
        Y = ds_cal.obs(self.obs)
        Z = np.log(np.maximum(Y, 1e-12))
        x_hat = self.point_estimate(Y, ds_cal.shaft_hz)
        Rlog = Z - self.core.predict_log(x_hat, ds_cal.shaft_hz)
        if not isinstance(self.residual, NullResidual):
            U = design_matrix(self.config.residual_inputs, shaft_hz=ds_cal.shaft_hz,
                              state=x_hat, z=Z, f_nom=self.spec.shaft_hz)
            Rlog = Rlog - self.residual.predict(U)
        if self.config.use_full_R:
            C = np.cov(Rlog, rowvar=False)
            C = np.atleast_2d(C)
            # Shrink toward the diagonal: with ~200 calibration frames and 9
            # channels the full covariance is usable, but the off-diagonals are
            # the part that decides the whitening and a noisy whitening makes
            # the innovation energy heavy-tailed. The same reasoning as the
            # Ledoit-Wolf shrinkage in pdmcore.models.anomaly.
            lam = 0.1
            self.R = (1.0 - lam) * C + lam * np.diag(np.diag(C))
        else:
            self.R = np.diag(np.var(Rlog, axis=0))
        self.R += 1e-12 * np.eye(self.R.shape[0])
        R_before = self.R.copy()

        # Adapt R so the estimator is consistent with its own innovations.
        #
        # R estimated above is the spread of the residual *after* the state was
        # fitted out of it, so it is biased low -- the fitted states absorb
        # n_state of the C channel degrees of freedom, and the filter's
        # innovation is taken at the predicted state, which carries the
        # prediction error too. Measured on the calibration window the filter
        # was over-confident by a factor of 3-4 in innovation energy, which
        # would make the whitening (and therefore the CUSUM's implicit
        # weighting of the channels) wrong even though the threshold is
        # standardised afterwards. One scalar inflation step, measured and
        # reported, rather than a tuned fudge.
        nis_ratio = 1.0
        if self.config.adapt_R:
            self.q_centre, self.q_scale = 0.0, 1.0
            nis = np.concatenate([
                self.run_trajectory(ds_cal.take(i).obs(self.obs), ds_cal.take(i).shaft_hz)["nis"]
                for i in self._traj_index(ds_cal)
            ])
            nis_ratio = float(np.median(nis) / max(1, self.R.shape[0]))
            if np.isfinite(nis_ratio) and nis_ratio > 0:
                self.R = self.R * nis_ratio

        # Standardise the deviation stream on the calibration window, then set
        # the threshold for the target ARL0 on that same stream.
        self.q_centre, self.q_scale = 0.0, 1.0
        q_cal = self._by_trajectory(ds_cal)
        self.q_centre, self.q_scale = robust_centre_scale(np.concatenate(q_cal))
        self.q_cal = [(q - self.q_centre) / self.q_scale for q in q_cal]
        self._cal_meta = {
            "q_centre": self.q_centre, "q_scale": self.q_scale,
            "R_cond": float(np.linalg.cond(self.R)),
            "R_inflation_from_nis": float(nis_ratio),
            "R_trace_before": float(np.trace(R_before)),
            "R_trace_after": float(np.trace(self.R)),
            "n_cal_frames": int(ds_cal.n),
            "filter_mode": self.config.filter_mode,
        }
        return self.set_target(target_arl0_frames)

    def set_target(self, target_arl0_frames: float) -> "HybridTwin":
        """Set the threshold for a target ARL0 on the cached calibration stream.

        Separated from :meth:`calibrate` because the matching target changes
        only the threshold: the innovation covariance, the standardisation and
        therefore the entire deviation stream are properties of the calibration
        window, not of the operating point. Recomputing them per target would
        cost three filter passes and would also let the two differ, which is the
        kind of coupling that makes an operating-point sweep quietly
        inconsistent.
        """
        if not hasattr(self, "q_cal"):
            raise RuntimeError("calibrate() must run before set_target()")
        base = CusumTest(drift=self.config.cusum_drift,
                         consecutive_required=self.config.consecutive_required)
        h, rep = threshold_for_arl0(self.q_cal, base, target_arl0_frames)
        self.test = CusumTest(drift=self.config.cusum_drift, threshold=h,
                              consecutive_required=self.config.consecutive_required)
        self.reports["calibration"] = {**rep, **self._cal_meta}
        return self

    # ------------------------------------------------------------------ 4 --
    def verify(self, ds_ver: Optional[Dataset] = None,
               q_ver: Optional[Sequence[np.ndarray]] = None) -> Dict[str, float]:
        """Realised alarm count on held-out healthy data vs a Poisson band.

        Accepts a precomputed deviation stream so an operating-point sweep does
        not re-filter the verification window once per threshold. The stream is
        a property of the window; only the count depends on the threshold.
        """
        if q_ver is None:
            if ds_ver is None:
                raise ValueError("verify needs ds_ver or q_ver")
            q_ver = self._by_trajectory(ds_ver)
        meas = measure_arl0(q_ver, self.test)
        expected = meas["n_frames"] / max(1e-9, self.reports["calibration"]["arl0_frames"])
        allowed = poisson_allowance(expected)
        out = {
            **meas,
            "expected_alarms": float(expected),
            "allowed_alarms_3sigma_poisson": float(allowed),
            "passed": bool(meas["n_alarms"] <= allowed),
        }
        self.reports["verification"] = out
        return out

    # -- convenience --------------------------------------------------------
    def fit(
        self, ds_fit: Dataset, ds_cal: Dataset, *, target_arl0_frames: float = 720.0,
        model: Optional[ObservationModel] = None,
    ) -> "HybridTwin":
        return (self.identify(ds_fit, model=model)
                .identify_process(ds_fit)
                .fit_residual(ds_fit)
                .calibrate(ds_fit, ds_cal, target_arl0_frames=target_arl0_frames))
