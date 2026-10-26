#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3 论文复现：Artificial Intelligence, Algorithmic Pricing, and Collusion
(AER, 2020, Calvano, Calzolari, Denicolò & Pastorello)

本脚本按论文公式重建独立 Q-learning 寡头定价实验，并生成缩减规模下的
参数网格、偏离-惩罚脉冲响应、策略相图等结果。

运行：
    python t3_qlearning.py --experiment all --workers 8
    python t3_qlearning.py --experiment quick --workers 4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/t3_mpl_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, root_scalar

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"
for _d in (FIG_DIR, RES_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 220,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})


# ---------------------------------------------------------------------------
# 1. 参数与模型
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelParams:
    n: int = 2
    m: int = 15
    mu: float = 0.25
    a0: float = 0.0
    xi: float = 0.10
    delta: float = 0.95
    c: tuple[float, ...] = (1.0, 1.0)
    a: tuple[float, ...] = (2.0, 2.0)

    def __post_init__(self) -> None:
        if len(self.c) != self.n or len(self.a) != self.n:
            raise ValueError("成本和质量向量必须与厂商数 n 一致")


@dataclass(frozen=True)
class SimParams:
    alpha: float
    beta: float
    max_steps: int = 1_500_000
    stable_window: int = 100_000
    policy_check: int = 5_000
    record_every: int = 0
    impulse_horizon: int = 0
    return_policy: bool = False
    convergence_eps: float = 0.001


class EconomicModel:
    """对称 logit 寡头，包含价格网格、收益张量和 Q 初始值。"""

    def __init__(self, p: ModelParams):
        self.p = p
        self.c = np.asarray(p.c, dtype=float)
        self.a = np.asarray(p.a, dtype=float)
        self.n = p.n
        self.m = p.m
        self.S = p.m ** p.n
        self.dims = (p.m,) * p.n
        self.grid_size = self.S
        self.pN, self.pM = self._solve_reference_prices()
        width = self.pM - self.pN
        self.prices = np.linspace(
            self.pN - p.xi * width,
            self.pM + p.xi * width,
            p.m,
        )
        self.profiles = np.column_stack(
            np.unravel_index(np.arange(self.S), self.dims)
        ).astype(np.int64)
        self.payoff, self.demand = self._build_payoff_tensors()
        self.payoff_flat = np.vstack([x.reshape(-1) for x in self.payoff])
        self.pi_nash = float(self.profit_at(np.full(self.n, self.pN))[0])
        self.pi_monopoly = float(self.profit_at(np.full(self.n, self.pM)).mean())
        axes = [j for j in range(self.n)]
        self.q0 = []
        for i in range(self.n):
            mean_axes = tuple(j for j in axes if j != i)
            self.q0.append(self.payoff[i].mean(axis=mean_axes))

    def demand_at(self, profile: np.ndarray) -> np.ndarray:
        profile = np.asarray(profile, dtype=float)
        x = np.r_[(self.a - profile) / self.p.mu, self.p.a0 / self.p.mu]
        x = x - x.max()
        e = np.exp(x)
        return e[: self.n] / e.sum()

    def profit_at(self, profile: np.ndarray) -> np.ndarray:
        profile = np.asarray(profile, dtype=float)
        return (profile - self.c) * self.demand_at(profile)

    def _best_response_symmetric(self, rival_price: float) -> float:
        def objective(own_price: float) -> float:
            profile = np.full(self.n, rival_price, dtype=float)
            profile[0] = own_price
            return -float(self.profit_at(profile)[0])

        upper = max(6.0, rival_price * 3.0)
        return float(minimize_scalar(
            objective, bounds=(0.05, upper), method="bounded"
        ).x)

    def _solve_reference_prices(self) -> tuple[float, float]:
        # 对称 Bertrand-Nash：价格等于对相同对手价格的最佳反应。
        pn = root_scalar(
            lambda x: self._best_response_symmetric(x) - x,
            bracket=(0.05, max(4.0, self.a.max() + 2.0)),
        ).root

        def neg_joint(price: float) -> float:
            return -float(self.profit_at(np.full(self.n, price)).sum())

        pm = minimize_scalar(
            neg_joint,
            bounds=(pn + 1e-8, max(12.0, self.a.max() + 8.0)),
            method="bounded",
        ).x
        if pm <= pn:
            raise RuntimeError("垄断价格未高于 Nash 价格，请检查需求参数")
        return float(pn), float(pm)

    def _build_payoff_tensors(self) -> tuple[list[np.ndarray], np.ndarray]:
        profiles = self.profiles
        price_profile = self.prices[profiles]
        x = np.column_stack([
            (self.a[None, :] - price_profile) / self.p.mu,
            np.full((self.S, 1), self.p.a0 / self.p.mu),
        ])
        x -= x.max(axis=1, keepdims=True)
        e = np.exp(x)
        dem = e / e.sum(axis=1, keepdims=True)
        inside = dem[:, : self.n]
        profits = (price_profile - self.c[None, :]) * inside
        payoffs = [
            profits[:, i].reshape(self.dims).astype(np.float64)
            for i in range(self.n)
        ]
        return payoffs, inside

    def profile_to_prices(self, profile_idx: int) -> np.ndarray:
        return self.prices[np.asarray(np.unravel_index(profile_idx, self.dims))]

    def static_best_response(
        self, rival_prices: np.ndarray, player: int = 0
    ) -> int:
        """给定其他厂商当前价格，选择自身静态最佳反应价格索引。"""
        rival_prices = np.asarray(rival_prices, dtype=float)
        values = []
        for ai, pi in enumerate(self.prices):
            prof = rival_prices.copy()
            prof[player] = pi
            values.append((pi - self.c[player]) * self.demand_at(prof)[player])
        return int(np.argmax(values))


def profile_index(actions: Iterable[int], m: int) -> int:
    out = 0
    for a in actions:
        out = out * m + int(a)
    return out


# ---------------------------------------------------------------------------
# 2. 独立 Q-learning 会话
# ---------------------------------------------------------------------------
def simulate_session(
    model_params: ModelParams,
    sim: SimParams,
    seed: int,
) -> dict[str, Any]:
    model = EconomicModel(model_params)
    rng = np.random.default_rng(seed)
    n, m, S = model.n, model.m, model.S
    Q = [np.tile(model.q0[i][None, :], (S, 1)).copy() for i in range(n)]
    state = int(rng.integers(S))
    policy = np.stack([np.argmax(q, axis=1) for q in Q])
    last_policy = policy.copy()
    stable = 0
    t = 0
    converged = False

    n_bins = math.ceil(sim.max_steps / sim.record_every) if sim.record_every else 0
    bin_sums = np.zeros(n_bins, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    while t < sim.max_steps:
        eps = math.exp(-sim.beta * t)
        actions = np.empty(n, dtype=np.int64)
        for i in range(n):
            if rng.random() < eps:
                actions[i] = int(rng.integers(m))
            else:
                actions[i] = int(np.argmax(Q[i][state]))

        pidx = profile_index(actions, m)
        rewards = model.payoff_flat[:, pidx]
        next_state = pidx
        for i in range(n):
            old = Q[i][state, actions[i]]
            target = rewards[i] + model_params.delta * float(Q[i][next_state].max())
            Q[i][state, actions[i]] = old + sim.alpha * (target - old)
        state = next_state

        if sim.record_every:
            denom = model.pi_monopoly - model.pi_nash
            delta_profit = (float(rewards.mean()) - model.pi_nash) / denom
            b = t // sim.record_every
            if b < n_bins:
                bin_sums[b] += delta_profit
                bin_counts[b] += 1

        t += 1
        if t % sim.policy_check == 0:
            policy = np.stack([np.argmax(q, axis=1) for q in Q])
            if np.array_equal(policy, last_policy):
                stable += sim.policy_check
            else:
                stable = 0
                last_policy = policy.copy()
            # 论文指出收敛只会发生在探索接近终止之后。仅检查策略稳定会
            # 在少数会话中过早停止，因此加入可审计的探索率门槛。
            if stable >= sim.stable_window and eps <= sim.convergence_eps:
                converged = True
                break

    policy = np.stack([np.argmax(q, axis=1) for q in Q]).astype(np.int32)
    cycle = deterministic_cycle(policy, state, m)
    cycle_rewards = np.array([
        model.payoff_flat[:, s].mean() for s in cycle
    ])
    delta_profit = float(
        (cycle_rewards.mean() - model.pi_nash)
        / (model.pi_monopoly - model.pi_nash)
    )
    cycle_prices = np.array([model.profile_to_prices(s) for s in cycle])
    long_run_prices = cycle_prices.mean(axis=0)

    nash_on_path = np.nan
    nash_all = np.nan
    qloss_on_path = np.nan
    qloss_all = np.nan
    if n == 2:
        best0, best1, ql0_all, ql1_all = best_responses_to_limit(model, policy)
        in_path = np.array(cycle, dtype=int)
        nash_on_path = bool(
            np.all(policy[0, in_path] == best0[in_path])
            and np.all(policy[1, in_path] == best1[in_path])
        )
        nash_all = bool(np.all(policy[0] == best0) and np.all(policy[1] == best1))
        qloss_on_path = float(np.mean([
            np.mean(ql0_all[in_path]), np.mean(ql1_all[in_path])
        ]))
        qloss_all = float(np.mean([ql0_all.mean(), ql1_all.mean()]))

    out: dict[str, Any] = {
        "seed": seed,
        "alpha": sim.alpha,
        "beta": sim.beta,
        "delta": model_params.delta,
        "n": n,
        "m": m,
        "steps": t,
        "converged": converged,
        "stable_periods": stable,
        "cycle_length": len(cycle),
        "delta_profit": delta_profit,
        "mean_price": float(long_run_prices.mean()),
        "price_gap": float(long_run_prices.max() - long_run_prices.min()),
        "nash_on_path": nash_on_path,
        "nash_all": nash_all,
        "qloss_on_path": qloss_on_path,
        "qloss_all": qloss_all,
        "pN": model.pN,
        "pM": model.pM,
    }
    out.update({
        f"price_{i+1}": float(long_run_prices[i]) for i in range(n)
    })

    if sim.record_every:
        expected = np.arange(len(bin_counts)) * sim.record_every
        keep = bin_counts > 0
        out["transition_time"] = expected[keep]
        out["transition_delta"] = bin_sums[keep] / bin_counts[keep]

    if sim.impulse_horizon:
        ir = impulse_response(model, policy, cycle, sim.impulse_horizon)
        out.update(ir)

    if sim.return_policy:
        out["policy"] = policy
        out["cycle"] = np.asarray(cycle, dtype=np.int32)
        out["current_state"] = state
        out["long_run_prices"] = long_run_prices
    return out


def deterministic_cycle(policy: np.ndarray, start: int, m: int) -> list[int]:
    seen: dict[int, int] = {}
    seq: list[int] = []
    s = int(start)
    while s not in seen:
        seen[s] = len(seq)
        seq.append(s)
        acts = policy[:, s]
        s = profile_index(acts, m)
        if len(seq) > 100_000:
            raise RuntimeError("未能找到有限状态策略的循环")
    return seq[seen[s]:]


def _policy_value_for_agent(
    model: EconomicModel,
    focal_policy: np.ndarray,
    rival_policy: np.ndarray,
    player: int = 0,
) -> np.ndarray:
    m, S, delta = model.m, model.S, model.p.delta
    rival = 1 - player
    a_f = focal_policy
    a_r = rival_policy
    nxt = a_f * m + a_r if player == 0 else a_r * m + a_f
    reward = model.payoff[player][a_f, a_r]
    P = np.zeros((S, S), dtype=np.float64)
    P[np.arange(S), nxt] = 1.0
    A = np.eye(S) - delta * P
    return np.linalg.solve(A, reward)


def best_responses_to_limit(
    model: EconomicModel,
    policy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """策略迭代：求每个厂商对对手极限策略的 Bellman 最优反应。"""
    if model.n != 2:
        raise ValueError("均衡检验仅实现双寡头情形")

    best: list[np.ndarray] = []
    losses: list[np.ndarray] = []
    for i in range(2):
        r = 1 - i
        focal = policy[i].copy()
        V_pi = _policy_value_for_agent(model, focal, policy[r], i)
        history: list[np.ndarray] = []
        for _ in range(100):
            history.append(focal.copy())
            V = _policy_value_for_agent(model, focal, policy[r], i)
            if i == 0:
                candidates = np.stack([
                    model.payoff[0][a, policy[1]] + model.p.delta * V[a * model.m + policy[1]]
                    for a in range(model.m)
                ])
            else:
                candidates = np.stack([
                    model.payoff[1][policy[0], a] + model.p.delta * V[policy[0] * model.m + a]
                    for a in range(model.m)
                ])
            improved = np.argmax(candidates, axis=0).astype(np.int32)
            if np.array_equal(improved, focal):
                break
            if any(np.array_equal(improved, h) for h in history[:-1]):
                break
            focal = improved
        V_opt = _policy_value_for_agent(model, focal, policy[r], i)
        rel_loss = np.maximum(0.0, (V_opt - V_pi) / np.maximum(np.abs(V_opt), 1e-12))
        best.append(focal)
        losses.append(rel_loss.astype(np.float64))
    return best[0], best[1], losses[0], losses[1]


def impulse_response(
    model: EconomicModel,
    policy: np.ndarray,
    cycle: list[int],
    horizon: int,
) -> dict[str, Any]:
    """从极限循环的每个节点出发，强制厂商 1 单期静态偏离。"""
    if model.n != 2:
        return {}
    all_dev = []
    all_non = []
    all_change_dev = []
    all_change_non = []
    cuts = []

    for start in cycle:
        pre_prices = model.profile_to_prices(start)
        path = np.empty((horizon, 2), dtype=float)
        path[0] = pre_prices
        rival_action = int(policy[1, start])
        # 厂商 1 对上一期对手价格做一次性静态最佳反应。
        dev_action = model.static_best_response(pre_prices, player=0)
        path[1] = model.prices[[dev_action, rival_action]]
        s = profile_index([dev_action, rival_action], model.m)
        for tau in range(2, horizon):
            acts = policy[:, s]
            path[tau] = model.prices[acts]
            s = profile_index(acts, model.m)
        all_dev.append(path[:, 0])
        all_non.append(path[:, 1])
        all_change_dev.append(path[:, 0] - pre_prices[0])
        all_change_non.append(path[:, 1] - pre_prices[1])
        if horizon > 2:
            cuts.append((path[2, 1] - pre_prices[1]) / pre_prices[1])

    return {
        "imp_dev_price": np.mean(all_dev, axis=0),
        "imp_non_price": np.mean(all_non, axis=0),
        "imp_dev_change": np.mean(all_change_dev, axis=0),
        "imp_non_change": np.mean(all_change_non, axis=0),
        "punishment_cut": float(np.mean(cuts)) if cuts else np.nan,
        "pre_price": float(np.mean([model.profile_to_prices(s).mean() for s in cycle])),
    }


# ---------------------------------------------------------------------------
# 3. 并行实验
# ---------------------------------------------------------------------------
def _grid_task(task: dict[str, Any]) -> dict[str, Any]:
    model_params, sim, seed = task["model"], task["sim"], task["seed"]
    res = simulate_session(model_params, sim, seed)
    res["grid_id"] = task["grid_id"]
    return res


def run_tasks(tasks: list[dict[str, Any]], workers: int, label: str) -> list[dict[str, Any]]:
    if workers <= 1:
        out = []
        for k, task in enumerate(tasks, 1):
            out.append(_grid_task(task))
            if k % max(1, len(tasks) // 10) == 0:
                print(f"[{label}] {k}/{len(tasks)}", flush=True)
        return out

    out = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_grid_task, task): task for task in tasks}
        for k, fut in enumerate(as_completed(futures), 1):
            out.append(fut.result())
            if k % max(1, len(tasks) // 10) == 0 or k == len(tasks):
                print(f"[{label}] {k}/{len(tasks)}  elapsed={time.time()-t0:.1f}s", flush=True)
    return out


def tasks_for_grid(
    alphas: list[float],
    betas: list[float],
    seeds: list[int],
    max_steps: int,
    stable_window: int,
) -> list[dict[str, Any]]:
    tasks = []
    for ia, alpha in enumerate(alphas):
        for ib, beta in enumerate(betas):
            for seed in seeds:
                tasks.append({
                    "grid_id": ia * len(betas) + ib,
                    "model": ModelParams(n=2, m=15, delta=0.95),
                    "sim": SimParams(
                        alpha=alpha,
                        beta=beta,
                        max_steps=max_steps,
                        stable_window=stable_window,
                        policy_check=5_000,
                        impulse_horizon=3,
                    ),
                    "seed": seed,
                })
    return tasks


def tasks_for_delta(
    deltas: list[float], seeds: list[int], max_steps: int, stable_window: int
) -> list[dict[str, Any]]:
    return [{
        "grid_id": i,
        "model": ModelParams(n=2, m=15, delta=delta),
        "sim": SimParams(
            alpha=0.15, beta=4e-6, max_steps=max_steps,
            stable_window=stable_window, policy_check=5_000,
        ),
        "seed": seed,
    } for i, delta in enumerate(deltas) for seed in seeds]


def tasks_for_players(
    players: list[int], seeds: list[int], max_steps: int, stable_window: int
) -> list[dict[str, Any]]:
    tasks = []
    for n in players:
        for seed in seeds:
            tasks.append({
                "grid_id": n,
                "model": ModelParams(
                    n=n, m=15, c=(1.0,) * n, a=(2.0,) * n, delta=0.95
                ),
                "sim": SimParams(
                    alpha=0.15, beta=4e-6, max_steps=max_steps,
                    stable_window=stable_window, policy_check=5_000,
                ),
                "seed": seed,
            })
    return tasks


def tasks_for_representative(
    seeds: list[int], max_steps: int, stable_window: int
) -> list[dict[str, Any]]:
    return [{
        "grid_id": seed,
        "model": ModelParams(n=2, m=15, delta=0.95),
        "sim": SimParams(
            alpha=0.15, beta=4e-6, max_steps=max_steps,
            stable_window=stable_window, policy_check=5_000,
            record_every=5_000, impulse_horizon=16, return_policy=True,
        ),
        "seed": seed,
    } for seed in seeds]


# ---------------------------------------------------------------------------
# 4. 结果保存
# ---------------------------------------------------------------------------
def scalar_frame(results: list[dict[str, Any]], drop: tuple[str, ...] = ()) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            k: v for k, v in r.items()
            if not isinstance(v, (np.ndarray, list, tuple)) and k not in drop
        })
    return pd.DataFrame(rows)


def save_intermediate(all_results: dict[str, list[dict[str, Any]]]) -> None:
    for name, results in all_results.items():
        scalar_frame(results).to_csv(RES_DIR / f"{name}_sessions.csv", index=False)


# ---------------------------------------------------------------------------
# 5. 绘图
# ---------------------------------------------------------------------------
def heatmap(
    df: pd.DataFrame,
    value: str,
    y_values: list[float],
    x_values: list[float],
    ylabel: str,
    xlabel: str,
    cbar_label: str,
    path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = ".2f",
) -> None:
    pivot = df.pivot_table(index="alpha", columns="beta", values=value, aggfunc="mean")
    mat = pivot.reindex(index=y_values, columns=x_values).to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = ax.imshow(
        mat, origin="lower", aspect="auto", cmap="viridis",
        vmin=vmin, vmax=vmax,
    )
    ax.set_yticks(np.arange(len(y_values)), [f"{v:g}" for v in y_values])
    ax.set_xticks(np.arange(len(x_values)), [f"{v*1e5:g}" for v in x_values])
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar_label)
    if len(y_values) <= 8 and len(x_values) <= 8:
        for i in range(len(y_values)):
            for j in range(len(x_values)):
                if np.isfinite(mat[i, j]):
                    ax.text(
                        j, i, format(mat[i, j], fmt), ha="center", va="center",
                        color="white" if mat[i, j] < np.nanmedian(mat) else "black",
                        fontsize=7,
                    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_fig1(df: pd.DataFrame, alphas: list[float], betas: list[float]) -> None:
    heatmap(
        df, "delta_profit", alphas, betas,
        r"Learning rate $\alpha$", r"Exploration decay $\beta\times10^5$",
        r"Average profit gain $\Delta$", FIG_DIR / "fig1_profit_gain_heatmap.png",
        vmin=0.0, vmax=1.0,
    )


def plot_fig2(df: pd.DataFrame, alphas: list[float], betas: list[float]) -> None:
    tmp = df.copy()
    tmp["equilibrium_share"] = tmp["nash_on_path"].astype(float)
    heatmap(
        tmp, "equilibrium_share", alphas, betas,
        r"Learning rate $\alpha$", r"Exploration decay $\beta\times10^5$",
        "Share of Nash-equilibrium sessions", FIG_DIR / "fig2_equilibrium_heatmap.png",
        vmin=0.0, vmax=1.0,
    )


def plot_fig7(df: pd.DataFrame, alphas: list[float], betas: list[float]) -> None:
    tmp = df.copy()
    tmp["punishment_pct"] = -100.0 * tmp["punishment_cut"]
    heatmap(
        tmp, "punishment_pct", alphas, betas,
        r"Learning rate $\alpha$", r"Exploration decay $\beta\times10^5$",
        "Nondeviator price cut at $\\tau=2$ (%)",
        FIG_DIR / "fig7_punishment_heatmap.png",
        vmin=0.0, vmax=max(20.0, float(tmp["punishment_pct"].quantile(0.95))),
    )


def plot_delta(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    g = df.groupby("delta")["delta_profit"]
    mean, q25, q75 = g.mean(), g.quantile(0.25), g.quantile(0.75)
    ax.fill_between(mean.index, q25, q75, alpha=0.20, color="#315b7d", label="IQR")
    ax.plot(mean.index, mean.values, marker="o", color="#102a43", lw=2.2, label="Mean")
    ax.axhline(0, color="0.55", lw=0.8, ls=":")
    ax.axhline(1, color="0.55", lw=0.8, ls=":")
    ax.set_xlabel(r"Discount factor $\delta$")
    ax.set_ylabel(r"Average profit gain $\Delta$")
    ax.set_ylim(-0.15, 1.05)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_discount_factor.png", bbox_inches="tight")
    plt.close(fig)


def _stack_impulse(results: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.vstack([np.asarray(r[key]) for r in results if key in r])


def plot_impulse(results: list[dict[str, Any]]) -> None:
    dev = _stack_impulse(results, "imp_dev_price")
    non = _stack_impulse(results, "imp_non_price")
    tau = np.arange(dev.shape[1])
    pN = float(np.mean([r["pN"] for r in results]))
    pM = float(np.mean([r["pM"] for r in results]))
    lr = float(np.mean([r["mean_price"] for r in results]))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(tau, dev.mean(axis=0), marker="s", ms=4, lw=1.8,
            color="#b23a48", label="Deviating agent")
    ax.plot(tau, non.mean(axis=0), marker="o", ms=4, lw=1.8,
            color="#1f6f8b", label="Nondeviating agent")
    ax.axhline(pN, color="0.35", ls="--", lw=1.0, label="Nash price")
    ax.axhline(pM, color="0.55", ls="-.", lw=1.0, label="Monopoly price")
    ax.axhline(lr, color="0.20", ls=":", lw=1.0, label="Long-run price")
    ax.set_xlabel(r"Time $\tau$")
    ax.set_ylabel("Price")
    ax.set_xticks(tau[::2])
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_impulse_response.png", bbox_inches="tight")
    plt.close(fig)

    devc = _stack_impulse(results, "imp_dev_change")
    nonc = _stack_impulse(results, "imp_non_change")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for ax, data, title in zip(
        axes, (devc, nonc), ("Deviating agent", "Nondeviating agent")
    ):
        x = np.arange(data.shape[1])
        lo, q25, mean, q75, hi = np.percentile(data, [5, 25, 50, 75, 95], axis=0)
        ax.fill_between(x, lo, hi, color="#9ebcda", alpha=0.20, label="5–95%")
        ax.fill_between(x, q25, q75, color="#4f7da1", alpha=0.38, label="IQR")
        ax.plot(x, mean, color="#102a43", lw=1.8, label="Mean")
        ax.axhline(0, color="0.5", lw=0.8, ls=":")
        ax.set_title(title)
        ax.set_xlabel(r"Time $\tau$")
        ax.set_xlim(0, data.shape[1] - 1)
    axes[0].set_ylabel("Price change")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_impulse_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_players(df: pd.DataFrame) -> None:
    g = df.groupby("n")["delta_profit"]
    mean, q25, q75 = g.mean(), g.quantile(0.25), g.quantile(0.75)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.bar(mean.index.astype(str), mean.values, color="#315b7d", alpha=0.85,
           yerr=[mean - q25, q75 - mean], capsize=4)
    for x, y in zip(mean.index.astype(str), mean.values):
        ax.text(x, y + 0.025, f"{y:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Number of firms")
    ax.set_ylabel(r"Average profit gain $\Delta$")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_players_robustness.png", bbox_inches="tight")
    plt.close(fig)


def plot_transition(results: list[dict[str, Any]]) -> None:
    valid = [r for r in results if "transition_time" in r]
    if not valid:
        return
    min_len = min(len(r["transition_time"]) for r in valid)
    t = np.asarray(valid[0]["transition_time"][:min_len])
    mat = np.vstack([np.asarray(r["transition_delta"][:min_len]) for r in valid])
    mean = mat.mean(axis=0)
    q25, q75 = np.percentile(mat, [25, 75], axis=0)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.fill_between(t, q25, q75, alpha=0.22, color="#4f7da1", label="IQR")
    ax.plot(t, mean, color="#102a43", lw=2.0, label="Mean")
    ax.set_xlabel("Periods (rolling 5,000-period mean)")
    ax.set_ylabel(r"Profit gain $\Delta$")
    ax.set_ylim(-0.1, 1.05)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig10_transition.png", bbox_inches="tight")
    plt.close(fig)


def plot_phase_portrait(result: dict[str, Any]) -> None:
    policy = np.asarray(result["policy"])
    m = int(result["m"])
    # 重新构造模型以获得完整价格网格和收益。
    model = EconomicModel(ModelParams(n=2, m=m, delta=float(result["delta"])))
    xx, yy = np.meshgrid(model.prices, model.prices, indexing="ij")
    xs = xx.reshape(-1)
    ys = yy.reshape(-1)
    nxt = np.array([
        profile_index(policy[:, s], m) for s in range(model.S)
    ], dtype=int)
    next_p = np.array([model.profile_to_prices(s) for s in nxt])
    dx = next_p[:, 0] - xs
    dy = next_p[:, 1] - ys
    # 缩短箭头，保留相位方向。
    scale = 0.20 / np.maximum(np.hypot(dx, dy), 1e-8)
    dx2, dy2 = dx * scale, dy * scale
    values = np.array([
        model.payoff_flat[:, s].mean() for s in nxt
    ])
    gain = (values - model.pi_nash) / (model.pi_monopoly - model.pi_nash)

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    sc = ax.scatter(xs, ys, c=gain, s=16, cmap="Greys", vmin=0, vmax=1,
                    edgecolor="none", alpha=0.8)
    ax.quiver(xs, ys, dx2, dy2, angles="xy", scale_units="xy", scale=1,
              color="#315b7d", alpha=0.45, width=0.0015)
    lr = np.asarray(result["long_run_prices"])
    ax.scatter(lr[0], lr[1], marker="s", s=90, facecolor="#b23a48",
               edgecolor="white", zorder=5, label="Long-run prices")
    ax.axhline(model.pN, color="0.5", lw=0.8, ls="--")
    ax.axvline(model.pN, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel("Previous price of firm 1")
    ax.set_ylabel("Previous price of firm 2")
    ax.set_xlim(model.prices[0] - 0.02, model.prices[-1] + 0.02)
    ax.set_ylim(model.prices[0] - 0.02, model.prices[-1] + 0.02)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Realized profit gain")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig9_phase_portrait.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. 主流程
# ---------------------------------------------------------------------------
def summarize(
    rep: list[dict[str, Any]],
    grid: list[dict[str, Any]],
    delta_res: list[dict[str, Any]],
    players: list[dict[str, Any]],
) -> dict[str, Any]:
    rep_df = pd.DataFrame(rep)
    grid_df = pd.DataFrame(grid)
    d_df = pd.DataFrame(delta_res)
    p_df = pd.DataFrame(players)
    summary = {
        "paper": "Artificial Intelligence, Algorithmic Pricing, and Collusion (AER 2020)",
        "scope": "reduced-scale reconstruction from the paper equations",
        "representative": {
            "sessions": int(len(rep_df)),
            "mean_delta_profit": float(rep_df["delta_profit"].mean()),
            "median_delta_profit": float(rep_df["delta_profit"].median()),
            "sd_delta_profit": float(rep_df["delta_profit"].std(ddof=1)),
            "mean_steps": float(rep_df["steps"].mean()),
            "converged_share": float(rep_df["converged"].mean()),
            "symmetric_fixed_share": float(
                ((rep_df["cycle_length"] == 1) & (rep_df["price_gap"] < 1e-9)).mean()
            ),
            "asymmetric_fixed_share": float(
                ((rep_df["cycle_length"] == 1) & (rep_df["price_gap"] >= 1e-9)).mean()
            ),
            "period2_cycle_share": float((rep_df["cycle_length"] == 2).mean()),
            "nash_on_path_share": float(rep_df["nash_on_path"].mean()),
            "mean_qloss_on_path": float(rep_df["qloss_on_path"].mean()),
            "mean_qloss_all": float(rep_df["qloss_all"].mean()),
            "mean_punishment_pct": float(-100 * rep_df["punishment_cut"].mean()),
            "share_sessions_with_price_cut": float((rep_df["punishment_cut"] < 0).mean()),
            "mean_long_run_price": float(rep_df["mean_price"].mean()),
            "pN": float(rep_df["pN"].mean()),
            "pM": float(rep_df["pM"].mean()),
        },
        "grid": {
            "cells": int(grid_df.groupby(["alpha", "beta"]).ngroups),
            "sessions": int(len(grid_df)),
            "mean_delta_profit": float(grid_df["delta_profit"].mean()),
            "delta_min": float(grid_df.groupby(["alpha", "beta"])["delta_profit"].mean().min()),
            "delta_max": float(grid_df.groupby(["alpha", "beta"])["delta_profit"].mean().max()),
            "corr_delta_punishment_harshness": float(
                -grid_df[["delta_profit", "punishment_cut"]].corr().iloc[0, 1]
            ),
        },
        "discount": {
            "sessions": int(len(d_df)),
            "mean_delta_by_delta": {
                f"{k:g}": float(v) for k, v in d_df.groupby("delta")["delta_profit"].mean().items()
            },
        },
        "players": {
            "sessions": int(len(p_df)),
            "mean_delta_by_n": {
                str(int(k)): float(v) for k, v in p_df.groupby("n")["delta_profit"].mean().items()
            },
        },
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["quick", "all"], default="all")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--replot", action="store_true", help="仅从已有 pickle 重画图")
    args = ap.parse_args()

    quick = args.experiment == "quick"
    if quick:
        alphas = [0.05, 0.15, 0.25]
        betas = np.array([0.5, 1.0, 2.0]) * 1e-5
        grid_seeds = [101]
        rep_seeds = [11, 12]
        deltas = [0.0, 0.5, 0.95]
        delta_seeds = [201]
        players = [2, 3, 4]
        player_seeds = [301]
        max_grid, max_rep, max_other = 500_000, 700_000, 500_000
        stable = 20_000
    else:
        alphas = [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25]
        betas = np.arange(0.25, 2.01, 0.25) * 1e-5
        grid_seeds = [1001, 1002, 1003, 1004]
        rep_seeds = list(range(2001, 2031))
        deltas = [0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.99]
        delta_seeds = list(range(3001, 3007))
        players = [2, 3, 4]
        player_seeds = list(range(4001, 4009))
        max_grid, max_rep, max_other = 2_500_000, 3_000_000, 2_500_000
        stable = 100_000

    pickled = RES_DIR / "all_results.pkl"
    if args.replot and pickled.exists():
        with pickled.open("rb") as f:
            all_results = pickle.load(f)
        rep = all_results["representative"]
        grid = all_results["grid"]
        delta_res = all_results["discount"]
        players_res = all_results["players"]
    else:
        print(f"Run mode={args.experiment}, workers={args.workers}", flush=True)
        rep = run_tasks(
            tasks_for_representative(rep_seeds, max_rep, stable),
            args.workers, "representative",
        )
        grid = run_tasks(
            tasks_for_grid(alphas, list(betas), grid_seeds, max_grid, stable),
            args.workers, "grid",
        )
        delta_res = run_tasks(
            tasks_for_delta(deltas, delta_seeds, max_other, stable),
            args.workers, "discount",
        )
        players_res = run_tasks(
            tasks_for_players(players, player_seeds, max_other, stable),
            args.workers, "players",
        )
        all_results = {
            "representative": rep,
            "grid": grid,
            "discount": delta_res,
            "players": players_res,
            "config": {
                "quick": quick,
                "alphas": alphas,
                "betas": list(map(float, betas)),
                "grid_seeds": grid_seeds,
                "rep_seeds": rep_seeds,
                "deltas": deltas,
                "delta_seeds": delta_seeds,
                "players": players,
                "player_seeds": player_seeds,
                "max_grid": max_grid,
                "max_rep": max_rep,
                "max_other": max_other,
                "stable_window": stable,
            },
        }
        with pickled.open("wb") as f:
            pickle.dump(all_results, f)
        save_intermediate({
            "representative": rep,
            "grid": grid,
            "discount": delta_res,
            "players": players_res,
        })

    # 汇总与图
    summary = summarize(rep, grid, delta_res, players_res)
    with (RES_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    grid_df = scalar_frame(grid)
    delta_df = scalar_frame(delta_res)
    player_df = scalar_frame(players_res)
    plot_fig1(grid_df, alphas, list(betas))
    plot_fig2(grid_df, alphas, list(betas))
    plot_fig7(grid_df, alphas, list(betas))
    plot_delta(delta_df)
    plot_impulse(rep)
    plot_transition(rep)
    plot_players(player_df)
    # 从固定点会话中选策略最接近代表性结果的一例画相图。
    fixed = [r for r in rep if r["cycle_length"] == 1]
    candidate = fixed[0] if fixed else rep[0]
    plot_phase_portrait(candidate)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Figures: {FIG_DIR}", flush=True)
    print(f"Results: {RES_DIR}", flush=True)


if __name__ == "__main__":
    main()
