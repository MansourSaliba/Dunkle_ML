"""
Phase II - SINDy on the ODE for water temperature dynamics

Target ODE:
    dTw/dt = f(I, Tw-Tg, Tw-Ta, Tg-Ta)

What this script does:
- Loads dunkle_clean.csv.
- Builds dTw/dt from finite differences of Tw over time_s.
- Uses a first-order feature library built from:
    [I, Tw-Tg, Tw-Ta, Tg-Ta]
- Performs STLSQ threshold sweep and selects the model by validation+sparsity.
- Uses a sequential split on derived ODE rows: 70% train, 20% val, 10% test.
- Evaluates selected model on train/val/test for dTw/dt.
- Integrates discovered ODE forward from Tw(0)=23.0 and compares Tw(t) trajectory.

Outputs:
- phase2_sindy_selection_dTw_dt.csv
- phase2_sindy_coefficients_dTw_dt.csv
- phase2_sindy_summary.csv
- phase2_sindy_rollout_Tw.csv
- phase2_sindy_rollout_Tw_plot.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRAIN_FRAC = 0.70
VAL_FRAC = 0.20
EPS = 1e-12
RIDGE = 1e-10
MAX_STLSQ_ITERS = 30
THRESHOLD_MIN_SCALE = 1e-4
THRESHOLD_MAX_SCALE = 1e-1
VAL_SPARSITY_WINDOW = 0.02
FEATURE_NAMES = ["I", "Tw-Tg", "Tw-Ta", "Tg-Ta"]


def load_data(base_dir: Path) -> pd.DataFrame:
    data_path = base_dir / "dunkle_clean.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing file: {data_path.name}")
    return pd.read_csv(data_path)


def get_column(df: pd.DataFrame, primary: str, fallback: str | None = None) -> np.ndarray:
    if primary in df.columns:
        return df[primary].to_numpy(dtype=float)
    if fallback is not None and fallback in df.columns:
        return df[fallback].to_numpy(dtype=float)
    raise ValueError(f"Missing required column: {primary}" + (f" (or {fallback})" if fallback else ""))


def split_indices(n: int) -> tuple[slice, slice, slice]:
    n_train = int(TRAIN_FRAC * n)
    n_val = int(VAL_FRAC * n)
    n_test = n - n_train - n_val
    return (
        slice(0, n_train),
        slice(n_train, n_train + n_val),
        slice(n_train + n_val, n_train + n_val + n_test),
    )


def build_ode_dataset(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    # Read features (supports both Ta and T_a naming).
    t = get_column(df, "time_s")
    I = get_column(df, "I")
    Ta = get_column(df, "Ta", fallback="T_a")
    Tw = get_column(df, "Tw")
    Tgi = get_column(df, "Tgi")

    if len(t) < 2:
        raise ValueError("Need at least 2 time points to compute finite-difference dTw/dt.")

    dt = np.diff(t)
    if np.any(dt <= 0.0):
        raise ValueError("time_s must be strictly increasing for finite differences.")

    dTw = np.diff(Tw)
    y = dTw / np.maximum(dt, EPS)

    # Forward-difference alignment: use state/features at time index i.
    X = np.column_stack(
        [
            I[:-1],
            Tw[:-1] - Tgi[:-1],
            Tw[:-1] - Ta[:-1],
            Tgi[:-1] - Ta[:-1],
        ]
    )
    return X, y, FEATURE_NAMES.copy()


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


def stlsq(X: np.ndarray, y: np.ndarray, threshold: float, max_iters: int = MAX_STLSQ_ITERS, ridge: float = RIDGE) -> np.ndarray:
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


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom <= 0.0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def select_threshold(records: list[dict[str, float]]) -> int:
    val_mse = np.array([r["val_mse"] for r in records], dtype=float)
    best = float(np.min(val_mse))
    cutoff = best * (1.0 + VAL_SPARSITY_WINDOW) if best > 0 else best + 1e-12

    eligible = [i for i, r in enumerate(records) if r["val_mse"] <= cutoff]
    eligible_sorted = sorted(eligible, key=lambda i: (records[i]["n_nonzero"], records[i]["val_mse"]))
    return eligible_sorted[0] if eligible_sorted else int(np.argmin(val_mse))


def format_equation(feature_names: list[str], coeffs: np.ndarray, target_name: str) -> str:
    active = np.where(np.abs(coeffs) > 0.0)[0]
    if len(active) == 0:
        return f"{target_name} = 0"

    parts = []
    for idx in active:
        c = coeffs[idx]
        name = feature_names[idx]
        if name == "1":
            parts.append(f"{c:.8e}")
        else:
            parts.append(f"({c:.8e})*{name}")
    return f"{target_name} = " + " + ".join(parts)


def rollout_tw(
    coeffs: np.ndarray,
    t: np.ndarray,
    I: np.ndarray,
    Ta: np.ndarray,
    Tgi: np.ndarray,
    tw0: float,
) -> np.ndarray:
    tw_pred = np.zeros_like(t, dtype=float)
    tw_pred[0] = float(tw0)
    for i in range(len(t) - 1):
        dt = float(t[i + 1] - t[i])
        if dt <= 0.0:
            raise ValueError("time_s must be strictly increasing for rollout integration.")

        x_i = np.array(
            [
                float(I[i]),
                float(tw_pred[i] - Tgi[i]),
                float(tw_pred[i] - Ta[i]),
                float(Tgi[i] - Ta[i]),
            ],
            dtype=float,
        )
        dTw_dt_i = float(x_i @ coeffs)
        tw_pred[i + 1] = tw_pred[i] + dt * dTw_dt_i

    return tw_pred


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    processed_dir = Path(__file__).resolve().parents[3] / "data" / "processed"
    df = load_data(processed_dir)

    X, y, feature_names = build_ode_dataset(df)
    train_slice, val_slice, test_slice = split_indices(len(y))

    X_train, X_val, X_test = X[train_slice], X[val_slice], X[test_slice]
    y_train, y_val, y_test = y[train_slice], y[val_slice], y[test_slice]

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

    coeff_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coeffs_best,
            "abs_coefficient": np.abs(coeffs_best),
            "is_active": np.abs(coeffs_best) > 0.0,
        }
    ).sort_values(["is_active", "abs_coefficient"], ascending=[False, False])

    selection_df = pd.DataFrame(records)

    coeff_path = base_dir / "phase2_sindy_coefficients_dTw_dt.csv"
    selection_path = base_dir / "phase2_sindy_selection_dTw_dt.csv"
    coeff_df.to_csv(coeff_path, index=False)
    selection_df.to_csv(selection_path, index=False)

    equation = format_equation(feature_names, coeffs_best, target_name="dTw_dt")

    # ODE rollout on full horizon from fixed initial condition Tw(0)=23.0.
    t_full = get_column(df, "time_s")
    I_full = get_column(df, "I")
    Ta_full = get_column(df, "Ta", fallback="T_a")
    Tw_true_full = get_column(df, "Tw")
    Tgi_full = get_column(df, "Tgi")

    tw0 = 23.0
    Tw_pred_full = rollout_tw(
        coeffs=coeffs_best,
        t=t_full,
        I=I_full,
        Ta=Ta_full,
        Tgi=Tgi_full,
        tw0=tw0,
    )

    # Rollout metrics over sequential regions (mapped to Tw rows).
    tw_train_slice = slice(0, train_slice.stop)
    tw_val_slice = slice(train_slice.stop, val_slice.stop)
    tw_test_slice = slice(val_slice.stop, len(t_full))

    tw_rollout_train_mse = mse(Tw_true_full[tw_train_slice], Tw_pred_full[tw_train_slice])
    tw_rollout_val_mse = mse(Tw_true_full[tw_val_slice], Tw_pred_full[tw_val_slice])
    tw_rollout_test_mse = mse(Tw_true_full[tw_test_slice], Tw_pred_full[tw_test_slice])

    tw_rollout_train_rmse = rmse(Tw_true_full[tw_train_slice], Tw_pred_full[tw_train_slice])
    tw_rollout_val_rmse = rmse(Tw_true_full[tw_val_slice], Tw_pred_full[tw_val_slice])
    tw_rollout_test_rmse = rmse(Tw_true_full[tw_test_slice], Tw_pred_full[tw_test_slice])

    tw_rollout_train_r2 = r2(Tw_true_full[tw_train_slice], Tw_pred_full[tw_train_slice])
    tw_rollout_val_r2 = r2(Tw_true_full[tw_val_slice], Tw_pred_full[tw_val_slice])
    tw_rollout_test_r2 = r2(Tw_true_full[tw_test_slice], Tw_pred_full[tw_test_slice])

    tw_rollout_rmse = rmse(Tw_true_full, Tw_pred_full)
    tw_rollout_r2 = r2(Tw_true_full, Tw_pred_full)

    rollout_df = pd.DataFrame(
        {
            "time_s": t_full,
            "Tw_true": Tw_true_full,
            "Tw_pred_rollout": Tw_pred_full,
        }
    )
    rollout_path = base_dir / "phase2_sindy_rollout_Tw.csv"
    rollout_df.to_csv(rollout_path, index=False)

    plt.figure(figsize=(10, 5.5))
    plt.plot(t_full, Tw_true_full, color="tab:blue", lw=2.0, label="Tw ground")
    plt.plot(t_full, Tw_pred_full, color="tab:red", lw=2.0, label="Tw predicted rollout")
    # Vertical split markers: train | val | test
    if 0 < train_slice.stop < len(t_full):
        plt.axvline(t_full[train_slice.stop], color="gray", lw=1.2, linestyle="--", label="train/val split")
    if 0 < val_slice.stop < len(t_full):
        plt.axvline(t_full[val_slice.stop], color="black", lw=1.2, linestyle=":", label="val/test split")
    plt.xlabel("time_s")
    plt.ylabel("Tw (degC)")
    plt.title("Phase II SINDy ODE Rollout: Tw Predicted vs Ground")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    rollout_plot_path = base_dir / "phase2_sindy_rollout_Tw_plot.png"
    plt.savefig(rollout_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    summary = {
        "target": "dTw_dt",
        "selected_threshold": best["threshold"],
        "n_nonzero": best["n_nonzero"],
        "train_mse": best["train_mse"],
        "val_mse": best["val_mse"],
        "test_mse": best["test_mse"],
        "train_rmse": best["train_rmse"],
        "val_rmse": best["val_rmse"],
        "test_rmse": best["test_rmse"],
        "train_r2": best["train_r2"],
        "val_r2": best["val_r2"],
        "test_r2": best["test_r2"],
        "rows_total_ode": int(len(y)),
        "train_points": int(len(y_train)),
        "val_points": int(len(y_val)),
        "test_points": int(len(y_test)),
        "test_evaluated": True,
        "rollout_tw0": tw0,
        "rollout_tw_rmse": tw_rollout_rmse,
        "rollout_tw_r2": tw_rollout_r2,
        "rollout_train_mse": tw_rollout_train_mse,
        "rollout_val_mse": tw_rollout_val_mse,
        "rollout_test_mse": tw_rollout_test_mse,
        "rollout_train_rmse": tw_rollout_train_rmse,
        "rollout_val_rmse": tw_rollout_val_rmse,
        "rollout_test_rmse": tw_rollout_test_rmse,
        "rollout_train_r2": tw_rollout_train_r2,
        "rollout_val_r2": tw_rollout_val_r2,
        "rollout_test_r2": tw_rollout_test_r2,
        "rollout_file": rollout_path.name,
        "rollout_plot_file": rollout_plot_path.name,
        "coefficients_file": coeff_path.name,
        "selection_file": selection_path.name,
        "equation": equation,
    }

    summary_df = pd.DataFrame([summary])
    summary_path = base_dir / "phase2_sindy_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("=" * 72)
    print("Phase II SINDy complete (ODE: dTw/dt)")
    print(f"Rows in derived ODE dataset: {len(y)}")
    print(f"Sequential split -> train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")
    print(f"Selected threshold: {best['threshold']:.6e}")
    print(f"Selected active terms: {best['n_nonzero']}")
    print(f"Test RMSE (selected model): {best['test_rmse']:.6e}")
    print(f"Test R2   (selected model): {best['test_r2']:.6f}")
    print(f"Rollout Test MSE  (Tw0=23.0): {tw_rollout_test_mse:.6e}")
    print(f"Rollout Tw RMSE (Tw0=23.0): {tw_rollout_rmse:.6e}")
    print(f"Rollout Tw R2   (Tw0=23.0): {tw_rollout_r2:.6f}")
    print(f"Selection saved: {selection_path.name}")
    print(f"Coefficients saved: {coeff_path.name}")
    print(f"Rollout saved: {rollout_path.name}")
    print(f"Rollout plot saved: {rollout_plot_path.name}")
    print(f"Summary saved: {summary_path.name}")
    print("=" * 72)


if __name__ == "__main__":
    main()
