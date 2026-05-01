"""
Phase II - SINDy on experimental data (multi-trajectory pooled fit)

Target ODE:
    dTw/dt = c1*Tg + c2*Tw + c3*Ta + c4*I

Data:
- Excel file with 11 day blocks.
- Each block has columns: Time, Ta, Tg, Tw, I
- Row 0 stores day label (e.g., "5-6 Jul"), row 1 stores headers.

Workflow:
- Keep each day as an independent trajectory.
- Split by day index: first 9 train, day 10 validation, day 11 test.
- Compute dTw/dt independently per day (never across day boundaries).
- Fit a single sparse coefficient vector on pooled train trajectories.
- Select threshold by validation MSE + sparsity preference.
- Track train/validation error across STLSQ iterations for selected model.
- Evaluate derivative fit (train/val/test).
- Roll out Tw forward day-by-day from each day's Tw(0).
- Save plots and one Excel report with coefficients and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Experimental workbook is restricted; request access before using it.
INPUT_FILE = Path("Experimental_Data_Cleaned_Final.xlsx")
OUTPUT_REPORT = Path("Phase_II_SINDy_exp_results.xlsx")
PLOTS_DIR = Path("Phase_II_SINDy_exp_plots")

COLUMNS_PER_DAY = 5  # Time, Ta, Tg, Tw, I
TRAIN_DAYS = 9
VAL_DAYS = 1
TEST_DAYS = 1

EPS = 1e-12
RIDGE = 1e-10
MAX_STLSQ_ITERS = 60
THRESHOLD_MIN_SCALE = 1e-4
THRESHOLD_MAX_SCALE = 1e-1
VAL_SPARSITY_WINDOW = 0.02
ITERATION_ERROR_PLOT = "sindy_train_val_error_vs_iteration.png"
ROLLOUT_TW_MIN = -20.0
ROLLOUT_TW_MAX = 120.0


@dataclass
class DayTrajectory:
    day_index: int
    day_label: str
    time_s: np.ndarray
    Ta: np.ndarray
    Tg: np.ndarray
    Tw: np.ndarray
    I: np.ndarray
    split: str


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom <= 0.0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def _parse_time_to_seconds(time_series: pd.Series) -> np.ndarray:
    # Try strict HH:MM:SS first, then flexible parsing.
    t1 = pd.to_datetime(time_series.astype(str), format="%H:%M:%S", errors="coerce")
    if t1.isna().any():
        t1 = pd.to_datetime(time_series.astype(str), errors="coerce")

    if t1.isna().any():
        # Fallback: assume 1-minute data if parsing fails.
        n = len(time_series)
        return np.arange(n, dtype=float) * 60.0

    sec = t1.dt.hour.to_numpy(dtype=float) * 3600.0
    sec += t1.dt.minute.to_numpy(dtype=float) * 60.0
    sec += t1.dt.second.to_numpy(dtype=float)

    # Build elapsed seconds relative to the first sample (typically ~08:00),
    # while handling a single midnight crossing. We only treat a large
    # backward jump as midnight; small backward jumps are treated as jitter.
    elapsed = np.zeros_like(sec, dtype=float)
    for i in range(1, len(sec)):
        delta = float(sec[i] - sec[i - 1])
        if delta < -12.0 * 3600.0:
            # True midnight rollover, e.g. 23:59 -> 00:00.
            delta += 24.0 * 3600.0
        elif delta < 0.0:
            # Non-physical small backward jump; keep monotonic timeline.
            delta = 0.0
        elapsed[i] = elapsed[i - 1] + delta
    return elapsed


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")


def _split_for_day(day_index: int, n_days: int) -> str:
    if day_index < TRAIN_DAYS:
        return "train"
    if day_index < TRAIN_DAYS + VAL_DAYS:
        return "val"
    if day_index < TRAIN_DAYS + VAL_DAYS + TEST_DAYS:
        return "test"
    raise ValueError(
        f"Unexpected day index {day_index + 1}. Expected exactly "
        f"{TRAIN_DAYS + VAL_DAYS + TEST_DAYS} days, found {n_days}."
    )


def load_day_trajectories(input_file: Path) -> list[DayTrajectory]:
    if not input_file.exists():
        raise FileNotFoundError(f"Missing file: {input_file}")

    df = pd.read_excel(input_file, sheet_name=0, header=None)

    if df.shape[1] % COLUMNS_PER_DAY != 0:
        raise ValueError(
            f"Expected columns in groups of {COLUMNS_PER_DAY} (Time, Ta, Tg, Tw, I). "
            f"Found {df.shape[1]} columns."
        )

    n_days = df.shape[1] // COLUMNS_PER_DAY
    expected_days = TRAIN_DAYS + VAL_DAYS + TEST_DAYS
    if n_days != expected_days:
        raise ValueError(f"Expected {expected_days} day blocks, found {n_days}.")

    trajectories: list[DayTrajectory] = []

    for day_idx, start_col in enumerate(range(0, df.shape[1], COLUMNS_PER_DAY)):
        day_label = str(df.iat[0, start_col]).strip()

        headers_raw = [str(df.iat[1, start_col + j]).strip() for j in range(COLUMNS_PER_DAY)]
        headers_norm = [h.rstrip() for h in headers_raw]

        block = df.iloc[2:, [start_col + j for j in range(COLUMNS_PER_DAY)]].copy()
        block.columns = headers_norm

        required = ["Time", "Ta", "Tg", "Tw"]
        i_col = "I" if "I" in block.columns else "I "
        if i_col not in block.columns:
            raise ValueError(f"Missing I column in day block {day_idx + 1} ({day_label}).")
        if any(col not in block.columns for col in required):
            raise ValueError(f"Missing required columns in day block {day_idx + 1} ({day_label}).")

        block["Ta"] = pd.to_numeric(block["Ta"], errors="coerce")
        block["Tg"] = pd.to_numeric(block["Tg"], errors="coerce")
        block["Tw"] = pd.to_numeric(block["Tw"], errors="coerce")
        block["I"] = pd.to_numeric(block[i_col], errors="coerce")

        # Keep rows where all required values exist.
        keep = block[["Time", "Ta", "Tg", "Tw", "I"]].notna().all(axis=1)
        block = block.loc[keep, ["Time", "Ta", "Tg", "Tw", "I"]].reset_index(drop=True)

        if len(block) < 2:
            raise ValueError(f"Day block {day_idx + 1} ({day_label}) has <2 valid rows.")

        time_s = _parse_time_to_seconds(block["Time"])
        dt = np.diff(time_s)
        if np.any(dt <= 0.0):
            raise ValueError(f"Non-increasing time in day block {day_idx + 1} ({day_label}).")

        trajectories.append(
            DayTrajectory(
                day_index=day_idx + 1,
                day_label=day_label,
                time_s=time_s.astype(float),
                Ta=block["Ta"].to_numpy(dtype=float),
                Tg=block["Tg"].to_numpy(dtype=float),
                Tw=block["Tw"].to_numpy(dtype=float),
                I=block["I"].to_numpy(dtype=float),
                split=_split_for_day(day_idx, n_days=n_days),
            )
        )

    return trajectories


def build_day_ode_arrays(day: DayTrajectory) -> tuple[np.ndarray, np.ndarray]:
    dt = np.diff(day.time_s)
    dTw = np.diff(day.Tw)
    y = dTw / np.maximum(dt, EPS)

    # Forward-difference alignment: use state/features at index i.
    X, _ = build_linear_feature_matrix(
        Ta=day.Ta[:-1],
        Tw=day.Tw[:-1],
        Tg=day.Tg[:-1],
        I=day.I[:-1],
    )
    return X, y


def build_linear_feature_matrix(
    Ta: np.ndarray,
    Tw: np.ndarray,
    Tg: np.ndarray,
    I: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    X = np.column_stack(
        [
            Tg,
            Tw,
            Ta,
            I,
        ]
    )
    names = [
        "Tg",
        "Tw",
        "Ta",
        "I",
    ]
    return X, names


def build_linear_feature_vector(ta: float, tw: float, tg: float, i_solar: float) -> np.ndarray:
    return np.array(
        [
            tg,
            tw,
            ta,
            i_solar,
        ],
        dtype=float,
    )


def solve_ridge(X: np.ndarray, y: np.ndarray, ridge: float = RIDGE) -> np.ndarray:
    xtx = X.T @ X
    rhs = X.T @ y
    eye = np.eye(xtx.shape[0])

    lam = max(float(ridge), 1e-10)
    for _ in range(8):
        try:
            return np.linalg.solve(xtx + lam * eye, rhs)
        except np.linalg.LinAlgError:
            lam *= 10.0
    return np.linalg.lstsq(xtx + lam * eye, rhs, rcond=None)[0]


def stlsq(
    X: np.ndarray,
    y: np.ndarray,
    threshold: float,
    max_iters: int = MAX_STLSQ_ITERS,
    ridge: float = RIDGE,
) -> np.ndarray:
    xi = solve_ridge(X, y, ridge=ridge)

    for _ in range(max_iters):
        small = np.abs(xi) < threshold
        xi_next = xi.copy()
        xi_next[small] = 0.0

        active = ~small
        if np.count_nonzero(active) == 0:
            return xi_next

        xi_refit = np.zeros_like(xi)
        xi_refit[active] = solve_ridge(X[:, active], y, ridge=ridge)

        if np.allclose(xi, xi_refit, rtol=1e-9, atol=1e-12):
            return xi_refit
        xi = xi_refit

    return xi


def stlsq_with_history(
    X: np.ndarray,
    y: np.ndarray,
    threshold: float,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_iters: int = MAX_STLSQ_ITERS,
    ridge: float = RIDGE,
) -> tuple[np.ndarray, pd.DataFrame]:
    xi = solve_ridge(X, y, ridge=ridge)
    history: list[dict[str, float | int]] = []

    def log_step(iter_idx: int, coeffs: np.ndarray) -> None:
        y_train_pred = X @ coeffs
        y_val_pred = X_val @ coeffs
        history.append(
            {
                "iteration": int(iter_idx),
                "n_nonzero": int(np.count_nonzero(np.abs(coeffs) > 0.0)),
                "train_mse": mse(y, y_train_pred),
                "val_mse": mse(y_val, y_val_pred),
                "train_rmse": rmse(y, y_train_pred),
                "val_rmse": rmse(y_val, y_val_pred),
            }
        )

    log_step(0, xi)

    for it in range(1, max_iters + 1):
        small = np.abs(xi) < threshold
        xi_next = xi.copy()
        xi_next[small] = 0.0

        active = ~small
        if np.count_nonzero(active) == 0:
            log_step(it, xi_next)
            return xi_next, pd.DataFrame(history)

        xi_refit = np.zeros_like(xi)
        xi_refit[active] = solve_ridge(X[:, active], y, ridge=ridge)
        log_step(it, xi_refit)

        if np.allclose(xi, xi_refit, rtol=1e-9, atol=1e-12):
            return xi_refit, pd.DataFrame(history)
        xi = xi_refit

    return xi, pd.DataFrame(history)


def select_threshold(records: list[dict[str, float]]) -> int:
    val_mse_all = np.array([r["val_mse"] for r in records], dtype=float)
    best = float(np.min(val_mse_all))
    cutoff = best * (1.0 + VAL_SPARSITY_WINDOW) if best > 0 else best + 1e-12
    eligible = [i for i, r in enumerate(records) if r["val_mse"] <= cutoff]
    eligible_sorted = sorted(eligible, key=lambda i: (records[i]["n_nonzero"], records[i]["val_mse"]))
    return eligible_sorted[0] if eligible_sorted else int(np.argmin(val_mse_all))


def rollout_tw(coeffs: np.ndarray, day: DayTrajectory) -> np.ndarray:
    tw_pred = np.zeros_like(day.Tw, dtype=float)
    tw_pred[0] = float(day.Tw[0])

    for i in range(len(day.Tw) - 1):
        dt = float(day.time_s[i + 1] - day.time_s[i])
        if dt <= 0.0:
            raise ValueError(f"Non-positive dt during rollout for day {day.day_index}.")

        tw_i = float(np.clip(tw_pred[i], ROLLOUT_TW_MIN, ROLLOUT_TW_MAX))
        x_i = build_linear_feature_vector(
            ta=float(day.Ta[i]),
            tw=tw_i,
            tg=float(day.Tg[i]),
            i_solar=float(day.I[i]),
        )
        dTw_dt_i = float(x_i @ coeffs)
        tw_next = tw_i + dt * dTw_dt_i
        tw_pred[i + 1] = float(np.clip(tw_next, ROLLOUT_TW_MIN, ROLLOUT_TW_MAX))

    return tw_pred


def format_equation(coeffs: np.ndarray) -> str:
    _, names = build_linear_feature_matrix(
        Ta=np.array([0.0], dtype=float),
        Tw=np.array([0.0], dtype=float),
        Tg=np.array([0.0], dtype=float),
        I=np.array([0.0], dtype=float),
    )
    parts = []
    for c, n in zip(coeffs, names):
        if abs(c) == 0.0:
            continue
        if n == "1":
            parts.append(f"{c:.8e}")
        else:
            parts.append(f"({c:.8e})*{n}")
    if not parts:
        return "dTw/dt = 0"
    return "dTw/dt = " + " + ".join(parts)


def _concat(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.array([], dtype=float)
    return np.concatenate(arrays, axis=0)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    processed_dir = Path(__file__).resolve().parents[3] / "data" / "processed"
    input_file = processed_dir / INPUT_FILE
    report_file = base_dir / OUTPUT_REPORT
    plots_dir = base_dir / PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    days = load_day_trajectories(input_file=input_file)

    # Build per-day ODE arrays independently (no cross-day derivative).
    day_ode = {}
    for d in days:
        Xd, yd = build_day_ode_arrays(d)
        day_ode[d.day_index] = (Xd, yd)

    train_days = [d for d in days if d.split == "train"]
    val_days = [d for d in days if d.split == "val"]
    test_days = [d for d in days if d.split == "test"]

    if len(train_days) != TRAIN_DAYS or len(val_days) != VAL_DAYS or len(test_days) != TEST_DAYS:
        raise RuntimeError("Unexpected split counts.")

    X_train = _concat([day_ode[d.day_index][0] for d in train_days])
    y_train = _concat([day_ode[d.day_index][1] for d in train_days])
    X_val = _concat([day_ode[d.day_index][0] for d in val_days])
    y_val = _concat([day_ode[d.day_index][1] for d in val_days])
    X_test = _concat([day_ode[d.day_index][0] for d in test_days])
    y_test = _concat([day_ode[d.day_index][1] for d in test_days])

    # Threshold sweep on pooled train trajectories (multi-trajectory equivalent).
    xi0 = solve_ridge(X_train, y_train)
    scale = max(float(np.max(np.abs(xi0))), 1e-6)
    thresholds = np.geomspace(scale * THRESHOLD_MIN_SCALE, scale * THRESHOLD_MAX_SCALE, num=20)

    records: list[dict[str, float]] = []
    coeff_bank: list[np.ndarray] = []

    for th in thresholds:
        coeffs = stlsq(X_train, y_train, threshold=float(th))
        coeff_bank.append(coeffs)

        y_train_pred = X_train @ coeffs
        y_val_pred = X_val @ coeffs
        y_test_pred = X_test @ coeffs

        records.append(
            {
                "threshold": float(th),
                "n_nonzero": int(np.count_nonzero(np.abs(coeffs) > 0.0)),
                "train_mse": mse(y_train, y_train_pred),
                "val_mse": mse(y_val, y_val_pred),
                "test_mse": mse(y_test, y_test_pred),
                "train_rmse": rmse(y_train, y_train_pred),
                "val_rmse": rmse(y_val, y_val_pred),
                "test_rmse": rmse(y_test, y_test_pred),
                "train_r2": r2(y_train, y_train_pred),
                "val_r2": r2(y_val, y_val_pred),
                "test_r2": r2(y_test, y_test_pred),
            }
        )

    best_idx = select_threshold(records)
    best = records[best_idx]
    coeffs_best = coeff_bank[best_idx]

    coeffs_hist, history_df = stlsq_with_history(
        X=X_train,
        y=y_train,
        threshold=float(best["threshold"]),
        X_val=X_val,
        y_val=y_val,
    )
    coeffs_best = coeffs_hist
    history_df["selected_threshold"] = float(best["threshold"])

    _, feature_names = build_linear_feature_matrix(
        Ta=np.array([0.0], dtype=float),
        Tw=np.array([0.0], dtype=float),
        Tg=np.array([0.0], dtype=float),
        I=np.array([0.0], dtype=float),
    )
    coeff_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coeffs_best,
            "abs_coefficient": np.abs(coeffs_best),
            "is_active": np.abs(coeffs_best) > 0.0,
        }
    ).sort_values(["is_active", "abs_coefficient"], ascending=[False, False])

    equation = format_equation(coeffs_best)

    err_plot_path = plots_dir / ITERATION_ERROR_PLOT
    plt.figure(figsize=(8.8, 4.8))
    plt.plot(history_df["iteration"], history_df["train_mse"], lw=2.0, label="Train MSE")
    plt.plot(history_df["iteration"], history_df["val_mse"], lw=2.0, label="Validation MSE")
    plt.xlabel("STLSQ iteration")
    plt.ylabel("MSE of dTw/dt")
    plt.title("SINDy Training Curve (Selected Threshold)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(err_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Rollout by day from measured initial Tw for each trajectory.
    rollout_rows: list[pd.DataFrame] = []
    rollout_metrics_per_day: list[dict[str, float | str | int]] = []

    for d in days:
        tw_pred = rollout_tw(coeffs_best, d)
        row = pd.DataFrame(
            {
                "day_index": d.day_index,
                "day_label": d.day_label,
                "split": d.split,
                "time_s": d.time_s,
                "Tw_true": d.Tw,
                "Tw_pred_rollout": tw_pred,
            }
        )
        rollout_rows.append(row)

        day_mse = mse(d.Tw, tw_pred)
        day_rmse = rmse(d.Tw, tw_pred)
        day_r2 = r2(d.Tw, tw_pred)
        rollout_metrics_per_day.append(
            {
                "day_index": d.day_index,
                "day_label": d.day_label,
                "split": d.split,
                "n_points": int(len(d.Tw)),
                "rollout_mse": day_mse,
                "rollout_rmse": day_rmse,
                "rollout_r2": day_r2,
            }
        )

        plt.figure(figsize=(10, 4.8))
        plt.plot(d.time_s / 3600.0, d.Tw, lw=2.0, color="tab:blue", label="Tw experimental")
        plt.plot(d.time_s / 3600.0, tw_pred, lw=2.0, color="tab:red", label="Tw predicted rollout")
        plt.xlabel("Elapsed time (hours)")
        plt.ylabel("Tw")
        plt.title(f"Day {d.day_index} ({d.day_label}) | Split: {d.split}")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        plot_path = plots_dir / f"{d.day_index:02d}_{_safe_name(d.day_label)}_Tw_rollout.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    rollout_all = pd.concat(rollout_rows, axis=0, ignore_index=True)
    rollout_per_day_df = pd.DataFrame(rollout_metrics_per_day)

    train_rollout = rollout_all.loc[rollout_all["split"] == "train"]
    val_rollout = rollout_all.loc[rollout_all["split"] == "val"]
    test_rollout = rollout_all.loc[rollout_all["split"] == "test"]

    rollout_train_mse = mse(
        train_rollout["Tw_true"].to_numpy(),
        train_rollout["Tw_pred_rollout"].to_numpy(),
    )
    rollout_val_mse = mse(
        val_rollout["Tw_true"].to_numpy(),
        val_rollout["Tw_pred_rollout"].to_numpy(),
    )
    rollout_test_mse = mse(
        test_rollout["Tw_true"].to_numpy(),
        test_rollout["Tw_pred_rollout"].to_numpy(),
    )
    rollout_train_rmse = rmse(
        train_rollout["Tw_true"].to_numpy(),
        train_rollout["Tw_pred_rollout"].to_numpy(),
    )
    rollout_val_rmse = rmse(
        val_rollout["Tw_true"].to_numpy(),
        val_rollout["Tw_pred_rollout"].to_numpy(),
    )
    rollout_test_rmse = rmse(
        test_rollout["Tw_true"].to_numpy(),
        test_rollout["Tw_pred_rollout"].to_numpy(),
    )
    rollout_train_r2 = r2(
        train_rollout["Tw_true"].to_numpy(),
        train_rollout["Tw_pred_rollout"].to_numpy(),
    )
    rollout_val_r2 = r2(
        val_rollout["Tw_true"].to_numpy(),
        val_rollout["Tw_pred_rollout"].to_numpy(),
    )
    rollout_test_r2 = r2(
        test_rollout["Tw_true"].to_numpy(),
        test_rollout["Tw_pred_rollout"].to_numpy(),
    )

    rollout_summary_df = pd.DataFrame(
        [
            {
                "rollout_train_mse": rollout_train_mse,
                "rollout_val_mse": rollout_val_mse,
                "rollout_test_mse": rollout_test_mse,
                "rollout_train_rmse": rollout_train_rmse,
                "rollout_val_rmse": rollout_val_rmse,
                "rollout_test_rmse": rollout_test_rmse,
                "rollout_train_r2": rollout_train_r2,
                "rollout_val_r2": rollout_val_r2,
                "rollout_test_r2": rollout_test_r2,
            }
        ]
    )

    derivative_summary_df = pd.DataFrame(
        [
            {
                "selected_threshold": best["threshold"],
                "n_nonzero_terms": best["n_nonzero"],
                "dTw_dt_train_mse": best["train_mse"],
                "dTw_dt_val_mse": best["val_mse"],
                "dTw_dt_test_mse": best["test_mse"],
                "dTw_dt_train_rmse": best["train_rmse"],
                "dTw_dt_val_rmse": best["val_rmse"],
                "dTw_dt_test_rmse": best["test_rmse"],
                "dTw_dt_train_r2": best["train_r2"],
                "dTw_dt_val_r2": best["val_r2"],
                "dTw_dt_test_r2": best["test_r2"],
                "rollout_train_mse": rollout_train_mse,
                "rollout_val_mse": rollout_val_mse,
                "rollout_test_mse": rollout_test_mse,
                "rollout_train_rmse": rollout_train_rmse,
                "rollout_val_rmse": rollout_val_rmse,
                "rollout_test_rmse": rollout_test_rmse,
                "rollout_train_r2": rollout_train_r2,
                "rollout_val_r2": rollout_val_r2,
                "rollout_test_r2": rollout_test_r2,
                "equation": equation,
                "input_file": INPUT_FILE.name,
                "plots_folder": PLOTS_DIR.as_posix(),
                "iteration_error_plot": err_plot_path.name,
                "split_by_day": "days 1-9 train, day 10 val, day 11 test",
                "note": "Per-day derivatives computed independently; pooled train fit with linear library [Tg, Tw, Ta, I].",
            }
        ]
    )

    split_df = pd.DataFrame(
        [
            {"day_index": d.day_index, "day_label": d.day_label, "split": d.split, "n_samples": len(d.Tw)}
            for d in days
        ]
    )

    threshold_df = pd.DataFrame(records)

    with pd.ExcelWriter(report_file, engine="openpyxl") as writer:
        derivative_summary_df.to_excel(writer, sheet_name="Summary", index=False)
        coeff_df.to_excel(writer, sheet_name="Coefficients", index=False)
        threshold_df.to_excel(writer, sheet_name="Threshold_Sweep", index=False)
        history_df.to_excel(writer, sheet_name="Training_History", index=False)
        split_df.to_excel(writer, sheet_name="Data_Split", index=False)
        rollout_summary_df.to_excel(writer, sheet_name="Rollout_Summary", index=False)
        rollout_per_day_df.to_excel(writer, sheet_name="Rollout_Per_Day", index=False)
        rollout_all.to_excel(writer, sheet_name="Rollout_All_Samples", index=False)

    print("=" * 78)
    print("Phase_II_SINDy_exp complete")
    print(f"Input: {INPUT_FILE.name}")
    print(f"Report: {OUTPUT_REPORT.name}")
    print(f"Plots: {PLOTS_DIR.as_posix()}/")
    print(f"Iteration error plot: {ITERATION_ERROR_PLOT}")
    print(f"Selected threshold: {best['threshold']:.6e}")
    print(f"Active terms: {best['n_nonzero']}")
    print(equation)
    print(f"dTw/dt MSE -> train: {best['train_mse']:.6e}, val: {best['val_mse']:.6e}, test: {best['test_mse']:.6e}")
    print(
        "Rollout MSE -> train: "
        f"{rollout_train_mse:.6e}, "
        f"val: {rollout_val_mse:.6e}, "
        f"test: {rollout_test_mse:.6e}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
