"""
Phase II - PySR on the ODE for water temperature dynamics

Target ODE:
    dTw/dt = f(I, Ta, Tg, Tw, 1)

What this script does:
- Loads dunkle_clean.csv.
- Builds dTw/dt from finite differences of Tw over time_s.
- Uses a first-order feature set:
    [I, Ta, Tg, Tw]
  with constants discovered by PySR.
- Uses a sequential split on derived ODE rows: 70% train, 20% val, 10% test.
- Trains PySR on train only and selects equation using validation error.
- Does not evaluate test error at this stage.

Outputs:
- phase2_pysr_training_report_dTw_dt.xlsx
- phase2_pysr_predictions_dTw_dt.xlsx
- phase2_pysr_rollout_Tw_plot.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from pysr import PySRRegressor
except ImportError as exc:
    raise ImportError(
        "PySR is required. Install with: pip install pysr"
    ) from exc


TRAIN_FRAC = 0.70
VAL_FRAC = 0.20
EPS = 1e-12

FEATURE_NAMES = ["I", "Ta", "Tg", "Tw"]
# PySR/SymPy reserves "I" as imaginary unit, so use a safe internal alias.
FEATURE_NAMES_PYSR = ["I_in", "Ta", "Tg", "Tw"]
TARGET_NAME = "dTw_dt"

# Moderate settings for non-extensive data.
NITERATIONS = 200
POPULATIONS = 10
POPULATION_SIZE = 40
MAXSIZE = 18
MAXDEPTH = 8
RANDOM_STATE = 42


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


def build_ode_dataset(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    t = get_column(df, "time_s")
    I = get_column(df, "I")
    Ta = get_column(df, "Ta", fallback="T_a")
    Tg = get_column(df, "Tg", fallback="Tgi")
    Tw = get_column(df, "Tw")

    if len(t) < 2:
        raise ValueError("Need at least 2 time points to compute finite-difference dTw/dt.")

    dt = np.diff(t)
    if np.any(dt <= 0.0):
        raise ValueError("time_s must be strictly increasing for finite differences.")

    y = np.diff(Tw) / np.maximum(dt, EPS)

    # Forward-difference alignment: use state/features at time index i.
    X = np.column_stack([I[:-1], Ta[:-1], Tg[:-1], Tw[:-1]])
    return X, y


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom <= 0.0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def safe_predict(model: PySRRegressor, X: np.ndarray, equation_index: int | None = None) -> np.ndarray:
    if equation_index is None:
        y_pred = model.predict(X)
    else:
        y_pred = model.predict(X, index=equation_index)
    return np.asarray(y_pred, dtype=float).reshape(-1)


def pick_best_equation_by_validation(model: PySRRegressor, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> tuple[int, pd.DataFrame]:
    equations_df = pd.DataFrame(model.equations_).copy()
    if equations_df.empty:
        raise RuntimeError("PySR returned no candidate equations.")

    records = []
    for idx in range(len(equations_df)):
        y_train_pred = safe_predict(model, X_train, equation_index=idx)
        y_val_pred = safe_predict(model, X_val, equation_index=idx)
        records.append(
            {
                "equation_index": idx,
                "train_mse": mse(y_train, y_train_pred),
                "val_mse": mse(y_val, y_val_pred),
                "train_rmse": rmse(y_train, y_train_pred),
                "val_rmse": rmse(y_val, y_val_pred),
                "train_r2": r2(y_train, y_train_pred),
                "val_r2": r2(y_val, y_val_pred),
            }
        )

    perf_df = pd.DataFrame(records)
    merged = equations_df.reset_index().rename(columns={"index": "equation_index"})
    merged = merged.merge(perf_df, on="equation_index", how="left")

    sort_cols = ["val_mse"]
    if "complexity" in merged.columns:
        sort_cols.append("complexity")
    merged = merged.sort_values(sort_cols, ascending=True).reset_index(drop=True)
    best_original_index = int(merged.loc[0, "equation_index"])
    return best_original_index, merged


def replace_internal_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "equation" in out.columns:
        out["equation_readable"] = (
            out["equation"]
            .astype(str)
            .str.replace("I_in", "I", regex=False)
        )
    if "sympy_format" in out.columns:
        out["sympy_readable"] = (
            out["sympy_format"]
            .astype(str)
            .str.replace("I_in", "I", regex=False)
        )
    return out


def rollout_tw_with_model(
    model: PySRRegressor,
    equation_index: int,
    t: np.ndarray,
    I: np.ndarray,
    Ta: np.ndarray,
    Tg: np.ndarray,
    tw0: float,
) -> np.ndarray:
    tw_pred = np.zeros_like(t, dtype=float)
    tw_pred[0] = float(tw0)

    for i in range(len(t) - 1):
        dt = float(t[i + 1] - t[i])
        if dt <= 0.0:
            raise ValueError("time_s must be strictly increasing for rollout integration.")

        x_i = np.array([[float(I[i]), float(Ta[i]), float(Tg[i]), float(tw_pred[i])]], dtype=float)
        dTw_dt_i = float(safe_predict(model, x_i, equation_index=equation_index)[0])
        tw_pred[i + 1] = tw_pred[i] + dt * dTw_dt_i

    return tw_pred


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    processed_dir = Path(__file__).resolve().parents[3] / "data" / "processed"
    df = load_data(processed_dir)

    X, y = build_ode_dataset(df)
    train_slice, val_slice, test_slice = split_indices(len(y))

    X_train, X_val = X[train_slice], X[val_slice]
    y_train, y_val = y[train_slice], y[val_slice]
    X_test, y_test = X[test_slice], y[test_slice]

    model = PySRRegressor(
        niterations=NITERATIONS,
        populations=POPULATIONS,
        population_size=POPULATION_SIZE,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[],
        model_selection="best",
        maxsize=MAXSIZE,
        maxdepth=MAXDEPTH,
        deterministic=True,
        random_state=RANDOM_STATE,
        procs=0,
        multithreading=False,
    )

    model.fit(X_train, y_train, variable_names=FEATURE_NAMES_PYSR)

    best_idx, equations_ranked_df = pick_best_equation_by_validation(model, X_train, y_train, X_val, y_val)
    equations_ranked_df = replace_internal_feature_names(equations_ranked_df)

    y_train_pred_best = safe_predict(model, X_train, equation_index=best_idx)
    y_val_pred_best = safe_predict(model, X_val, equation_index=best_idx)
    y_test_pred_best = safe_predict(model, X_test, equation_index=best_idx)

    # ODE rollout using the selected discovered equation.
    t_full = get_column(df, "time_s")
    I_full = get_column(df, "I")
    Ta_full = get_column(df, "Ta", fallback="T_a")
    Tg_full = get_column(df, "Tg", fallback="Tgi")
    Tw_true_full = get_column(df, "Tw")
    tw0 = float(Tw_true_full[0])

    Tw_pred_full = rollout_tw_with_model(
        model=model,
        equation_index=best_idx,
        t=t_full,
        I=I_full,
        Ta=Ta_full,
        Tg=Tg_full,
        tw0=tw0,
    )

    tw_train_slice = slice(0, train_slice.stop)
    tw_val_slice = slice(train_slice.stop, val_slice.stop)
    tw_test_slice = slice(val_slice.stop, len(t_full))
    rollout_train_mse = mse(Tw_true_full[tw_train_slice], Tw_pred_full[tw_train_slice])
    rollout_val_mse = mse(Tw_true_full[tw_val_slice], Tw_pred_full[tw_val_slice])
    rollout_test_mse = mse(Tw_true_full[tw_test_slice], Tw_pred_full[tw_test_slice])

    best_metrics = {
        "target": TARGET_NAME,
        "selected_equation_index": best_idx,
        "train_mse": mse(y_train, y_train_pred_best),
        "val_mse": mse(y_val, y_val_pred_best),
        "test_mse": mse(y_test, y_test_pred_best),
        "train_rmse": rmse(y_train, y_train_pred_best),
        "val_rmse": rmse(y_val, y_val_pred_best),
        "test_rmse": rmse(y_test, y_test_pred_best),
        "train_r2": r2(y_train, y_train_pred_best),
        "val_r2": r2(y_val, y_val_pred_best),
        "test_r2": r2(y_test, y_test_pred_best),
        "rows_total_ode": int(len(y)),
        "train_points": int(len(y_train)),
        "val_points": int(len(y_val)),
        "test_points": int(len(y_test)),
        "test_evaluated": True,
        "rollout_tw0": tw0,
        "rollout_train_mse": rollout_train_mse,
        "rollout_val_mse": rollout_val_mse,
        "rollout_test_mse": rollout_test_mse,
    }

    run_config = {
        "algorithm": "PySR",
        "target": TARGET_NAME,
        "features": ", ".join(FEATURE_NAMES),
        "features_internal_for_pysr": ", ".join(FEATURE_NAMES_PYSR),
        "constant_term": "enabled (PySR constants)",
        "binary_operators": "+, -, *, /",
        "unary_operators": "none",
        "train_fraction": TRAIN_FRAC,
        "val_fraction": VAL_FRAC,
        "test_fraction": 1.0 - TRAIN_FRAC - VAL_FRAC,
        "split_type": "sequential",
        "niterations": NITERATIONS,
        "populations": POPULATIONS,
        "population_size": POPULATION_SIZE,
        "maxsize": MAXSIZE,
        "maxdepth": MAXDEPTH,
        "deterministic": True,
        "random_state": RANDOM_STATE,
        "note": "Test set evaluated on best equation selected by validation.",
    }

    prediction_train_df = pd.DataFrame(
        {
            "row_index": np.arange(train_slice.start, train_slice.stop),
            "y_true_dTw_dt": y_train,
            "y_pred_dTw_dt": y_train_pred_best,
        }
    )
    prediction_val_df = pd.DataFrame(
        {
            "row_index": np.arange(val_slice.start, val_slice.stop),
            "y_true_dTw_dt": y_val,
            "y_pred_dTw_dt": y_val_pred_best,
        }
    )
    prediction_test_df = pd.DataFrame(
        {
            "row_index": np.arange(test_slice.start, test_slice.stop),
            "y_true_dTw_dt": y_test,
            "y_pred_dTw_dt": y_test_pred_best,
        }
    )
    rollout_df = pd.DataFrame(
        {
            "time_s": t_full,
            "Tw_true": Tw_true_full,
            "Tw_pred_rollout": Tw_pred_full,
        }
    )

    split_labels = np.full(len(t_full), "test", dtype=object)
    split_labels[tw_train_slice] = "train"
    split_labels[tw_val_slice] = "val"
    rollout_df["split"] = split_labels

    training_report_path = base_dir / "phase2_pysr_training_report_dTw_dt.xlsx"
    predictions_path = base_dir / "phase2_pysr_predictions_dTw_dt.xlsx"
    rollout_plot_path = base_dir / "phase2_pysr_rollout_Tw_plot.png"

    with pd.ExcelWriter(training_report_path) as writer:
        pd.DataFrame([run_config]).to_excel(writer, sheet_name="run_config", index=False)
        pd.DataFrame([best_metrics]).to_excel(writer, sheet_name="best_model", index=False)
        equations_ranked_df.to_excel(writer, sheet_name="equations_ranked", index=False)
        pd.DataFrame(
            [
                {
                    "train_start": train_slice.start,
                    "train_end_exclusive": train_slice.stop,
                    "val_start": val_slice.start,
                    "val_end_exclusive": val_slice.stop,
                    "test_start": test_slice.start,
                    "test_end_exclusive": test_slice.stop,
                }
            ]
        ).to_excel(writer, sheet_name="split_indices", index=False)

    with pd.ExcelWriter(predictions_path) as writer:
        prediction_train_df.to_excel(writer, sheet_name="train_predictions", index=False)
        prediction_val_df.to_excel(writer, sheet_name="val_predictions", index=False)
        prediction_test_df.to_excel(writer, sheet_name="test_predictions", index=False)
        rollout_df.to_excel(writer, sheet_name="rollout_tw", index=False)

    plt.figure(figsize=(10, 5.5))
    plt.plot(t_full, Tw_true_full, color="tab:blue", lw=2.0, label="Tw true")
    plt.plot(t_full, Tw_pred_full, color="tab:red", lw=2.0, label="Tw predicted rollout")
    if 0 < train_slice.stop < len(t_full):
        plt.axvline(t_full[train_slice.stop], color="gray", lw=1.2, linestyle="--", label="train/val split")
    if 0 < val_slice.stop < len(t_full):
        plt.axvline(t_full[val_slice.stop], color="black", lw=1.2, linestyle=":", label="val/test split")
    plt.xlabel("time_s")
    plt.ylabel("Tw (degC)")
    plt.title("Phase II PySR ODE Rollout: Tw Predicted vs True")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(rollout_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("=" * 72)
    print("Phase II PySR complete (ODE: dTw/dt)")
    print(f"Rows in derived ODE dataset: {len(y)}")
    print(f"Sequential split -> train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")
    print(f"Selected equation index (validation-based): {best_idx}")
    print(f"\nPoint-wise derivative predictions (dTw/dt):")
    print(f"  Train MSE: {mse(y_train, y_train_pred_best):.6e}, RMSE: {rmse(y_train, y_train_pred_best):.6e}")
    print(f"  Val   MSE: {mse(y_val, y_val_pred_best):.6e}, RMSE: {rmse(y_val, y_val_pred_best):.6e}")
    print(f"  Test  MSE: {mse(y_test, y_test_pred_best):.6e}, RMSE: {rmse(y_test, y_test_pred_best):.6e}")
    print(f"\nODE rollout (Tw integration):")
    print(f"  Train MSE: {rollout_train_mse:.6e}")
    print(f"  Val   MSE: {rollout_val_mse:.6e}")
    print(f"  Test  MSE: {rollout_test_mse:.6e}")
    print(f"Training report saved: {training_report_path.name}")
    print(f"Predictions saved: {predictions_path.name}")
    print(f"Rollout plot saved: {rollout_plot_path.name}")
    print("=" * 72)


if __name__ == "__main__":
    main()
