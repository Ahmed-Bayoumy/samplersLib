# pylint: disable=too-many-lines
"""
# ------------------------------------------------------------------------------------#
#  Samplers Library - (SamplersLib)                                                   #
#                                                                                     #
#  Author: Ahmed H. Bayoumy                                                           #
#  email: ahmed.bayoumy@mail.mcgill.ca                                                #
#                                                                                     #
#  This program is free software: you can redistribute it and/or modify it under the  #
#  terms of the GNU General Public License as published by the Free Software          #
#  Foundation, either version 3 of the License, or (at your option) any later         #
#  version.                                                                           #
#                                                                                     #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY    #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    #
#  PARTICULAR PURPOSE.  See the GNU General Public License for more details.          #
#                                                                                     #
#  You should have received a copy of the GNU General Public License along            #
#  with this program. If not, see <http://www.gnu.org/licenses/>.                     #
#                                                                                     #
#  You can find information on SamplersLib at                                         #
#  https://github.com/Ahmed-Bayoumy/samplersLib                                       #
#  Copyright (C) 2024-2026  Ahmed H. Bayoumy                                          #
# ------------------------------------------------------------------------------------#
"""

import math
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import qmc

from .samplers import BiTPE


def _require_int(value, minimum: int, message: str) -> int:
    """`value` as an int, or ValueError(`message`) if it is not an integer >= `minimum`."""
    if int(value) != value or value < minimum:
        raise ValueError(message)
    return int(value)


class CholeskyKDE:
    """
    Gaussian kernel density estimate with a full bandwidth matrix, used only
    through its Cholesky factor.

    The bandwidth matrix is ``H = h^2 S``, where ``S`` is the covariance of
    `data` shrunk toward its diagonal (Ledoit-Wolf) and ``h = n^(-1/(d+4))``
    (Scott's rule). ``H + jitter I = L L^T`` is factored once; log-densities
    use triangular solves with ``L`` and samples are ``x_j + L eps``, so ``H``
    is never inverted. Shrinkage keeps ``S`` positive definite even when there
    are fewer points than dimensions, where the plain sample covariance is
    singular.
    """

    JITTER = 1e-9

    def __init__(self, data: np.ndarray):
        from sklearn.covariance import ledoit_wolf

        self.data = np.atleast_2d(np.asarray(data, dtype=float))
        n, d = self.data.shape
        S = ledoit_wolf(self.data)[0] if n > 1 else np.eye(d) * 1e-2
        H = n ** (-2.0 / (d + 4)) * S
        scale = np.trace(H) / d
        self.L = np.linalg.cholesky(H + self.JITTER * max(scale, 1e-12) * np.eye(d))
        self.log_norm = -np.log(np.diag(self.L)).sum() - 0.5 * d * np.log(2 * np.pi) - np.log(n)

    def log_pdf(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        diff = (X[:, None, :] - self.data[None, :, :]).reshape(-1, X.shape[1])
        z = solve_triangular(self.L, diff.T, lower=True)
        q = -0.5 * (z**2).sum(axis=0).reshape(X.shape[0], self.data.shape[0])
        return np.logaddexp.reduce(q, axis=1) + self.log_norm


class ThompsonAllocator:
    """
    Discounted Thompson sampling over two arms.

    Each arm keeps a Beta(a, b) posterior on its success probability. Every
    update first discounts all evidence toward the Beta(1, 1) prior by
    `discount`, so the allocation can follow changes during a run; with
    probability `p_min` the arm is chosen uniformly at random, so neither arm
    is ever shut out.
    """

    def __init__(self, rng: np.random.Generator, p_min: float = 0.1, discount: float = 0.98, n_arms: int = 2):
        self.rng = rng
        self.p_min = p_min
        self.discount = discount
        self.ab = np.ones((n_arms, 2))

    def choose(self) -> int:
        draws = self.rng.beta(self.ab[:, 0], self.ab[:, 1])
        arm = int(np.argmax(draws))
        return arm if self.rng.random() >= self.p_min else int(self.rng.integers(len(self.ab)))

    def update(self, arm: int, reward: float):
        self.ab = 1.0 + self.discount * (self.ab - 1.0)
        self.ab[arm] += (reward, 1.0 - reward)


class AdaptiveHybridTPE:
    """
    Adaptive hybrid of a global reduced-space TPE and a model-free local
    trust region, for box-constrained minimization.

    Every evaluation is proposed by one of two arms, chosen by discounted
    Thompson sampling on whether the arm's points improve the best value:

    - global (arm 1): `BiTPE` fitted to all evaluated points over the whole
      box, with a `n_reduced_dims`-dimensional reduction and the chosen
      `lift`; the candidate with the largest l/g is proposed. By default
      ``n_reduced_dims = 3`` for ``D <= 20`` and ``ceil(sqrt(D))`` above.
    - local (arm 0): a trust region around the current iterate ``x_k`` with
      half-width ``delta`` (as a fraction of each variable's range).
      Candidates are drawn around ``x_k`` with the step metric ``C = diag(d)
      + U U^T`` learned from accepted local steps and kept across iterations
      (``d`` is a running mean of squared normalized steps, ``U`` holds the
      `metric_rank` leading directions of the recent accepted steps and of
      their evolution path), scaled so that the largest coordinate standard
      deviation equals ``delta``, and clipped to the box. The candidate with
      the largest l/g (full-space Cholesky KDEs of the good and bad sets) is
      proposed. A step is accepted when it achieves sufficient decrease,
      ``f < f_k - sufficient_decrease * delta^2 * std(f)``. ``delta`` follows
      the success rate of the local steps (the 1/5th success rule of the
      (1+1)-ES): a running success rate ``p`` (smoothing ``p_t / (2 + p_t)``,
      target ``p_t = 0.2``) updates ``delta <- delta * exp((p - p_t) / (d_s (1
      - p_t)))`` after every local step, with damping ``d_s`` =
      `success_damping`, capped at `delta_max`, so ``delta`` shrinks in
      proportion to the distance to a local minimum. The default ``d_s = 2``
      adapts the radius within the few hundred evaluations of an expensive
      problem; ``success_damping=None`` gives the slower ``1 + D / 2`` of the
      (1+1)-ES, which reached lower regret after 2000 evaluations on BBOB. When ``delta`` falls below `delta_min` the region
      restarts at `delta_init` around the best point, and the metric and the
      success rate are reset.

    A global proposal that improves on ``f_k`` re-centres the trust region
    there and keeps ``delta``.

    `n_candidates` sets the aggression of both arms: each proposal is the
    best of `n_candidates` scored candidates, so a larger pool makes the
    choice greedier (more exploitation around the good points) and a smaller
    pool leaves more to chance (more exploration). With ``n_candidates=1``
    the l/g score plays no part and each arm samples its proposal directly.

    With ``uniform_prob = rho > 0``, each post-design proposal is, with
    probability ``rho`` and independently of everything else, a uniform point
    of the box (handled like a global proposal). This safeguard makes the best
    value converge almost surely to the global minimum of any continuous
    objective, whatever the two arms do; the default ``rho = 0`` keeps the
    unguarded method.

    With ``halton_every = m > 0``, every m-th post-design evaluation is instead
    the next point of the unscrambled Halton sequence (handled like a global
    proposal), on a fixed schedule; Thompson sampling chooses the arm in all
    other iterations. The Halton points are dense in the box, so the best value
    converges to the global minimum of any continuous objective on every run,
    for any initial design. The default ``m = 0`` disables it.

    The following options select earlier variants of the local arm and of the
    re-centring; together (``recenter_delta="reset"``, ``local_metric="kde"``,
    ``step_control="tr"``, ``local_center="mixed"``, ``n_reduced_dims=3``)
    they reproduce the first version of the method.

    - ``recenter_delta="reset"``: a global improvement re-centres the trust
      region with ``delta = delta_init``.
    - ``local_metric="kde"``: local steps are shaped by the Cholesky factor of
      the good set's `CholeskyKDE` instead of the learned metric.
    - ``step_control="tr"``: ``delta`` doubles on an accepted step that
      reaches the edge of the region and halves after `fail_tol` consecutive
      rejections.
    - ``local_center="mixed"``: half of the local candidates are drawn around
      good points inside the trust region instead of around ``x_k``.
    - ``decrease_scale="local"`` measures sufficient decrease against the
      spread of f over the points inside the trust region (or over the good
      set when fewer than three lie inside) instead of over all evaluated
      points.
    - ``restart="diverse"``: when ``delta`` falls below `delta_min`, the
      trust region restarts at the best point at least ``delta_init / 2``
      (in the max-norm of the unit cube) from the collapsed centre, or at the
      best point if there is none.

    Use `ask`/`tell` for an external evaluation loop, or `minimize` for a
    callable. The first `n_init` points are a Latin hypercube design (or
    `initial_points`).
    """

    def __init__(
        self,
        vlim: np.ndarray,
        n_init: int = 20,
        seed: int = 0,
        gamma: float = 0.25,
        max_good: int = 25,
        n_candidates: int = 8,
        n_reduced_dims: Optional[int] = None,
        reduction_method: str = "pca",
        lift: str = "residual",
        kernel_type: Optional[Dict[str, float]] = None,
        delta_init: float = 0.4,
        delta_max: float = 0.5,
        delta_min: float = 1e-3,
        fail_tol: int = 3,
        sufficient_decrease: float = 1e-3,
        p_min: float = 0.1,
        discount: float = 0.98,
        initial_points: Optional[np.ndarray] = None,
        uniform_prob: float = 0.0,
        halton_every: int = 0,
        decrease_scale: str = "global",
        recenter_delta: str = "keep",
        local_metric: str = "lowrank",
        metric_rank: int = 3,
        restart: str = "best",
        step_control: str = "success",
        local_center: str = "current",
        success_damping: Optional[float] = 2.0,
    ):
        self.vlim = np.asarray(vlim, dtype=float)
        if self.vlim.ndim != 2 or self.vlim.shape[1] != 2 or np.any(self.vlim[:, 1] <= self.vlim[:, 0]):
            raise ValueError("`vlim` must have shape (D, 2) with lower < upper bound in every row.")
        self.dim = self.vlim.shape[0]
        self.seed = seed
        self.gamma, self.max_good = gamma, max_good
        self.n_candidates = _require_int(n_candidates, 1, "`n_candidates` must be a positive integer.")
        if n_reduced_dims is None:
            n_reduced_dims = 3 if self.dim <= 20 else math.ceil(math.sqrt(self.dim))
        self.n_reduced_dims = _require_int(n_reduced_dims, 1, "`n_reduced_dims` must be a positive integer or None.")
        self.reduction_method, self.lift = reduction_method, lift
        self.kernel_type = {"Gaussian": 1.0} if kernel_type is None else dict(kernel_type)
        self.delta_init, self.delta_max, self.delta_min = delta_init, delta_max, delta_min
        self.fail_tol, self.sufficient_decrease = fail_tol, sufficient_decrease

        self._init_U = self._initial_design(n_init, initial_points)

        if not 0.0 <= uniform_prob <= 1.0:
            raise ValueError("`uniform_prob` must lie in [0, 1].")
        self.uniform_prob = float(uniform_prob)
        # separate stream, consulted only when uniform_prob > 0, so the default
        # behaviour is unchanged
        self._safeguard_rng = np.random.default_rng(seed + 37)
        self.safeguard_steps = 0
        self.halton_every = _require_int(halton_every, 0, "`halton_every` must be a nonnegative integer (0 disables it).")
        # unscrambled Halton sequence: deterministic and dense in the unit cube;
        # its first point (the origin) is skipped
        self._halton = qmc.Halton(d=self.dim, scramble=False)
        self._halton.fast_forward(1)
        for name, value, allowed in (
            ("decrease_scale", decrease_scale, ("global", "local")),
            ("recenter_delta", recenter_delta, ("reset", "keep")),
            ("local_metric", local_metric, ("kde", "lowrank")),
            ("restart", restart, ("best", "diverse")),
            ("step_control", step_control, ("tr", "success")),
            ("local_center", local_center, ("mixed", "current")),
        ):
            if value not in allowed:
                raise ValueError(f"`{name}` must be one of {allowed}.")
        self.metric_rank = _require_int(metric_rank, 1, "`metric_rank` must be a positive integer.")
        self.decrease_scale, self.recenter_delta = decrease_scale, recenter_delta
        self.local_metric, self.restart = local_metric, restart
        self.step_control, self.local_center = step_control, local_center
        if success_damping is not None and success_damping <= 0:
            raise ValueError("`success_damping` must be positive (or None for 1 + D / 2).")
        self.success_damping = 1 + self.dim / 2 if success_damping is None else float(success_damping)
        self._p_succ = self.SUCCESS_TARGET
        self._reset_metric()
        self.allocator = ThompsonAllocator(np.random.default_rng(seed + 23), p_min=p_min, discount=discount)
        self._fallback_rng = np.random.default_rng(seed + 29)
        self._U = np.empty((0, self.dim))
        self._F = np.empty(0)
        self._pending: Optional[Tuple[np.ndarray, Optional[int]]] = None
        self._it = 0
        self._xk = self._fk = None
        self.delta = delta_init
        self._n_fail = 0
        self.restarts = 0
        self.failures = 0
        self.arms = []  # arm of every post-design evaluation: 0 local, 1 global

    def _initial_design(self, n_init: int, initial_points: Optional[np.ndarray]) -> np.ndarray:
        """Initial design in unit-cube coordinates: `initial_points`, or a Latin hypercube of `n_init` points."""
        if initial_points is None:
            if n_init < 3:
                raise ValueError("`n_init` must be at least 3 (two good points and one bad point).")
            return qmc.LatinHypercube(d=self.dim, seed=np.random.default_rng(self.seed)).random(n_init)
        pts = np.atleast_2d(np.asarray(initial_points, dtype=float))
        if pts.shape[1] != self.dim or len(pts) < 3:
            raise ValueError("`initial_points` must have shape (n >= 3, D).")
        return self._to_unit(pts)

    # ------------------------------------------------------------------ data
    def _to_unit(self, X):
        return (np.asarray(X, dtype=float) - self.vlim[:, 0]) / (self.vlim[:, 1] - self.vlim[:, 0])

    def _from_unit(self, U):
        return self.vlim[:, 0] + np.asarray(U, dtype=float) * (self.vlim[:, 1] - self.vlim[:, 0])

    @property
    def X(self) -> np.ndarray:
        """Evaluated points, in original units."""
        return self._from_unit(self._U)

    @property
    def F(self) -> np.ndarray:
        """Objective values of `X`."""
        return self._F.copy()

    @property
    def best(self) -> Tuple[np.ndarray, float]:
        """Best evaluated point and its value."""
        if not len(self._F):
            raise RuntimeError("No evaluations yet.")
        i = int(np.argmin(self._F))
        return self._from_unit(self._U[i]), float(self._F[i])

    def _split(self):
        order = np.argsort(self._F)
        n_good = int(min(max(2, math.ceil(self.gamma * len(self._F))), self.max_good))
        good, bad = order[:n_good], order[n_good:]
        return self._U[good], self._F[good], self._U[bad], self._F[bad]

    # ------------------------------------------------------------ proposals
    def _propose_global(self) -> np.ndarray:
        gU, gF, bU, bF = self._split()
        tpe = BiTPE(
            good_data=gU,
            good_f_values=gF,
            bad_data=bU,
            bad_f_values=bF,
            kernel_type=self.kernel_type,
            n_r=self.n_candidates,
            vlim=np.array([[0.0, 1.0]] * self.dim),
            seed=100_000 * self.seed + self._it,
            n_reduced_dims=min(self.n_reduced_dims, self.dim),
            reduction_method=self.reduction_method,
            lift=self.lift,
        )
        cands = tpe.resample()
        return cands[int(np.argmax(tpe.candidate_ratios))]

    def _propose_local(self) -> np.ndarray:
        rng = np.random.default_rng(100_000 * self.seed + self._it + 5)
        xk, delta = self._xk, self.delta
        lo, hi = np.clip(xk - delta, 0, 1), np.clip(xk + delta, 0, 1)
        gU, _, bU, _ = self._split()
        l_kde, g_kde = CholeskyKDE(gU), CholeskyKDE(bU)
        # step shape: the good set's Cholesky factor, or the learned metric,
        # scaled so its largest coordinate standard deviation spans delta
        L = l_kde.L if self.local_metric == "kde" else self._metric_factor()
        L = L / max(np.sqrt((L**2).sum(axis=1)).max(), 1e-12) * delta
        n_half = self.n_candidates // 2 if self.local_center == "mixed" else self.n_candidates
        around_center = xk + rng.standard_normal((n_half, L.shape[1])) @ L.T
        inside = gU[np.all((gU >= lo) & (gU <= hi), axis=1)]
        n_rest = self.n_candidates - n_half
        bases = inside[rng.integers(len(inside), size=n_rest)] if len(inside) else np.repeat(xk[None], n_rest, 0)
        around_good = bases + rng.standard_normal((len(bases), L.shape[1])) @ L.T
        X = np.clip(np.vstack([around_center, around_good]), lo, hi)
        score = np.nan_to_num(l_kde.log_pdf(X) - g_kde.log_pdf(X), nan=-np.inf)
        return X[int(np.argmax(score))] if np.isfinite(score).any() else X[0]

    # ---------------------------------------------------------- local metric
    METRIC_LR_DIAG = 0.2
    METRIC_LR_PATH = 0.3

    def _reset_metric(self):
        self._metric_d = np.ones(self.dim)
        self._metric_path = np.zeros(self.dim)
        self._metric_steps = []

    def _metric_factor(self) -> np.ndarray:
        """Factor ``A`` with ``A A^T = diag(d) + U U^T``, shape (D, D + rank)."""
        d = np.maximum(self._metric_d, 1e-6 * self._metric_d.max())
        A = np.diag(np.sqrt(d))
        rows = self._metric_steps + ([self._metric_path] if np.any(self._metric_path) else [])
        if len(rows) >= 2:
            M = np.asarray(rows) / np.sqrt(len(rows))
            _, sv, Vt = np.linalg.svd(M, full_matrices=False)
            k = min(self.metric_rank, len(sv))
            A = np.hstack([A, Vt[:k].T * sv[:k]])
        return A

    def _update_metric(self, s: np.ndarray):
        """Learn from an accepted local step `s`, normalized by the radius."""
        cd, cp = self.METRIC_LR_DIAG, self.METRIC_LR_PATH
        self._metric_d = (1 - cd) * self._metric_d + cd * s**2
        self._metric_path = (1 - cp) * self._metric_path + np.sqrt(cp * (2 - cp)) * s
        self._metric_steps = (self._metric_steps + [s])[-max(2 * self.metric_rank, 10):]

    # -------------------------------------------------------------- ask/tell
    def ask(self) -> np.ndarray:
        """Next point to evaluate, in original units."""
        if self._pending is not None:
            return self._from_unit(self._pending[0])
        n = len(self._F)
        if n < len(self._init_U):
            self._pending = (self._init_U[n], None)
            return self._from_unit(self._init_U[n])
        if self.halton_every and (n - len(self._init_U) + 1) % self.halton_every == 0:
            # deterministic safeguard: the next point of the Halton sequence on a
            # fixed schedule, handled as a global proposal
            self.safeguard_steps += 1
            u = self._halton.random(1)[0]
            self._pending = (u, 1)
            return self._from_unit(u)
        if self.uniform_prob > 0 and self._safeguard_rng.random() < self.uniform_prob:
            # convergence safeguard: a uniform point, handled as a global proposal
            self.safeguard_steps += 1
            u = self._safeguard_rng.random(self.dim)
            self._pending = (u, 1)
            return self._from_unit(u)
        arm = self.allocator.choose()
        try:
            u = self._propose_global() if arm == 1 else self._propose_local()
            if not np.all(np.isfinite(u)):
                raise FloatingPointError("non-finite proposal")
        except Exception:  # pylint: disable=broad-except
            self.failures += 1
            if arm == 1:
                u = self._fallback_rng.random(self.dim)
            else:
                u = np.clip(self._xk + self._fallback_rng.uniform(-self.delta, self.delta, self.dim), 0, 1)
        self._pending = (u, arm)
        return self._from_unit(u)

    def tell(self, x: np.ndarray, f: float):
        """Report the value `f` of the point returned by the last `ask`."""
        if self._pending is None:
            raise RuntimeError("`tell` must follow `ask`.")
        u, arm = self._pending
        if not np.allclose(self._to_unit(x), u, atol=1e-9):
            raise ValueError("`x` does not match the point returned by `ask`.")
        f = float(f)
        if not np.isfinite(f):
            raise ValueError("The objective value must be finite.")
        self._pending = None

        if arm is None:  # initial design
            self._U = np.vstack([self._U, u])
            self._F = np.append(self._F, f)
            if len(self._F) == len(self._init_U):
                k = int(np.argmin(self._F))
                self._xk, self._fk = self._U[k].copy(), self._F[k]
            return

        self.allocator.update(arm, float(f < self._F.min()))
        self.arms.append(arm)
        if arm == 1:
            if f < self._fk:
                self._xk, self._fk, self._n_fail = u.copy(), f, 0
                if self.recenter_delta == "reset":
                    self.delta = self.delta_init
            self._U = np.vstack([self._U, u])
            self._F = np.append(self._F, f)
        else:
            self._update_trust_region(u, f)
        self._it += 1

    SUCCESS_TARGET = 0.2

    def _update_trust_region(self, u: np.ndarray, f: float):
        """Sufficient-decrease test and radius update after a local step."""
        success = f < self._fk - self.sufficient_decrease * self.delta**2 * max(self._decrease_scale(), 1e-12)
        if success:
            at_edge = np.max(np.abs(u - self._xk)) >= 0.99 * self.delta
            if self.local_metric == "lowrank":
                self._update_metric((u - self._xk) / self.delta)
            self._xk, self._fk, self._n_fail = u.copy(), f, 0
            if at_edge and self.step_control == "tr":
                self.delta = min(2 * self.delta, self.delta_max)
        elif self.step_control == "tr":
            self._n_fail += 1
            if self._n_fail >= self.fail_tol:
                self.delta, self._n_fail = self.delta / 2, 0
        if self.step_control == "success":
            pt = self.SUCCESS_TARGET
            self._p_succ += pt / (2 + pt) * (float(success) - self._p_succ)
            self.delta = min(self.delta * math.exp((self._p_succ - pt) / (self.success_damping * (1 - pt))), self.delta_max)
        self._U = np.vstack([self._U, u])
        self._F = np.append(self._F, f)
        if self.delta < self.delta_min:
            F = self._F
            if self.restart == "diverse":
                far = np.max(np.abs(self._U - self._xk), axis=1) >= self.delta_init / 2
                F = np.where(far, self._F, np.inf) if far.any() else self._F
            k = int(np.argmin(F))
            self._xk, self._fk = self._U[k].copy(), self._F[k]
            self.delta, self.restarts = self.delta_init, self.restarts + 1
            self._p_succ = self.SUCCESS_TARGET
            if self.local_metric == "lowrank":
                self._reset_metric()

    def _decrease_scale(self) -> float:
        """Spread of f that the sufficient-decrease test is measured against."""
        if self.decrease_scale == "global":
            return self._F.std()
        inside = np.all(np.abs(self._U - self._xk) <= self.delta, axis=1)
        if inside.sum() >= 3:
            return self._F[inside].std()
        return self._split()[1].std()

    def minimize(self, func: Callable[[np.ndarray], float], budget: int) -> Tuple[np.ndarray, float]:
        """Evaluate `func` until `budget` evaluations in total; return the best point and value."""
        while len(self._F) < budget:
            x = self.ask()
            self.tell(x, func(x))
        return self.best
