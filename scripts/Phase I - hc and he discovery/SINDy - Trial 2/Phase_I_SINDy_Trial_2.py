"""
Phase I - Sparse Regression (STLSQ) Trial 2

Changes vs previous SINDy run:
- Keep same sequential split/evaluation workflow (70/20/10, evaluate train/val/test).
- Remove physics-derived compound term from hc library.
- Remove hc from he library.
- Use ONLY first-order polynomial terms of remaining base variables.

Outputs (trial 2 specific):
- sindy_trial2_selection_hc.csv
- sindy_trial2_selection_he.csv
- sindy_trial2_coefficients_hc.csv
- sindy_trial2_coefficients_he.csv
- sindy_trial2_loss_progress_hc.png
- sindy_trial2_loss_progress_he.png
- sindy_trial2_forward_predictions_hc.csv
- sindy_trial2_forward_predictions_he.csv
- sindy_trial2_forward_overlay_hc.png
- sindy_trial2_forward_overlay_he.png
- phase1_sindy_trial2_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TRAIN_FRAC = 0.70
VAL_FRAC = 0.20
EPS = 1e-12
MAX_STLSQ_ITERS = 30
RIDGE = 1e-10


def load_data(base_dir: Path) -> pd.DataFrame:
    data_path = base_dir / "dunkle_clean.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing file: {data_path.name}")
    return pd.read_csv(data_path)


def validate_columns(df: pd.DataFrame) -> None:
    required = {"Tw", "Tgi", "Pw", "Pgi", "hc", "he"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def split_indices(n: int) -> tuple[slice, slice, slice]:
    n_train = int(TRAIN_FRAC * n)
    n_val = int(VAL_FRAC * n)
    n_test = n - n_train - n_val
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n_train + n_val + n_test)


def safe_divide(num: np.ndarray, den: np.ndarray, eps: float = EPS) -> np.ndarray:
    out = np.zeros_like(num, dtype=float)
    mask = np.abs(den) > eps
    out[mask] = num[mask] / den[mask]
    return out


def add_first_order_only(base: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Build ONLY first-order polynomial terms + intercept."""
    names = list(base.keys())
    feature_names = ["1"] + names
    X = np.column_stack([np.ones(len(next(iter(base.values()))), dtype=float)] + [base[n] for n in names])
    return X, feature_names


def build_library_hc(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    Tw = df["Tw"].to_numpy(dtype=float)
    Tgi = df["Tgi"].to_numpy(dtype=float)
    Pw = df["Pw"].to_numpy(dtype=float)
    Pgi = df["Pgi"].to_numpy(dtype=float)

    dT = Tw - Tgi
    dP = Pw - Pgi

    # Trial 2: no compound term.
    base = {
        "Tw": Tw,
        "Tgi": Tgi,
        "Pw": Pw,
        "Pgi": Pgi,
        "dT": dT,
        "dP": dP,
        "dP_over_dT": safe_divide(dP, dT),
        "dT_over_dP": safe_divide(dT, dP),
    }

    return add_first_order_only(base)


def build_library_he(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    Tw = df["Tw"].to_numpy(dtype=float)
    Tgi = df["Tgi"].to_numpy(dtype=float)
    Pw = df["Pw"].to_numpy(dtype=float)
    Pgi = df["Pgi"].to_numpy(dtype=float)

    dT = Tw - Tgi
    dP = Pw - Pgi

    # Trial 2: no hc feature.
    base = {
        "Tw": Tw,
        "Tgi": Tgi,
        "Pw": Pw,
        "Pgi": Pgi,
        "dT": dT,
        "dP": dP,
        "dP_over_dT": safe_divide(dP, dT),
        "dT_over_dP": safe_divide(dT, dP),
    }

    return add_first_order_only(base)


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


def select_threshold(records: list[dict[str, float]]) -> int:
    val_mse = np.array([r["val_mse"] for r in records], dtype=float)
    best = float(np.min(val_mse))
    cutoff = best * 1.02 if best > 0 else best + 1e-12

    eligible = [i for i, r in enumerate(records) if r["val_mse"] <= cutoff]
    eligible_sorted = sorted(eligible, key=lambda i: (records[i]["n_nonzero"], records[i]["val_mse"]))
    return eligible_sorted[0] if eligible_sorted else int(np.argmin(val_mse))


def run_sparse_regression(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    train_slice: slice,
    val_slice: slice,
    test_slice: slice,
    out_dir: Path,
) -> tuple[dict[str, float], np.ndarray]:
    X_train, X_val, X_test = X[train_slice], X[val_slice], X[test_slice]
    y_train, y_val, y_test = y[train_slice], y[val_slice], y[test_slice]

    xi0 = solve_ridge(X_train, y_train)
    scale = float(np.max(np.abs(xi0)))
    scale = max(scale, 1e-6)
    thresholds = np.geomspace(scale * 1e-6, scale * 1e-1, num=20)

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

    coeff_path = out_dir / f"sindy_trial2_coefficients_{name}.csv"
    selection_path = out_dir / f"sindy_trial2_selection_{name}.csv"
    coeff_df.to_csv(coeff_path, index=False)
    selection_df.to_csv(selection_path, index=False)

    plot_path = out_dir / f"sindy_trial2_loss_progress_{name}.png"
    x = np.arange(len(selection_df))
    plt.figure(figsize=(9, 5))
    plt.plot(x, selection_df["train_mse"].to_numpy(dtype=float), lw=1.8, label="train_mse")
    plt.plot(x, selection_df["val_mse"].to_numpy(dtype=float), lw=1.8, label="val_mse")
    plt.xlabel("Threshold sweep index")
    plt.ylabel("MSE")
    plt.title(f"STLSQ Trial 2 Loss Progression - {name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    equation = format_equation(feature_names, coeffs_best, target_name=name)

    summary = {
        "target": name,
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
        "test_points": int(len(y_test)),
        "coefficients_file": coeff_path.name,
        "selection_file": selection_path.name,
        "loss_plot_file": plot_path.name,
        "equation": equation,
    }
    return summary, coeffs_best


def evaluate_forward_predictions(
    target_name: str,
    X_full: np.ndarray,
    y_full: np.ndarray,
    coeffs: np.ndarray,
    df: pd.DataFrame,
    train_slice: slice,
    val_slice: slice,
    test_slice: slice,
    out_dir: Path,
) -> dict[str, float | str]:
    y_pred = X_full @ coeffs

    forward_metrics: dict[str, float | str] = {
        f"{target_name}_forward_train_mse": mse(y_full[train_slice], y_pred[train_slice]),
        f"{target_name}_forward_val_mse": mse(y_full[val_slice], y_pred[val_slice]),
        f"{target_name}_forward_test_mse": mse(y_full[test_slice], y_pred[test_slice]),
        f"{target_name}_forward_train_rmse": rmse(y_full[train_slice], y_pred[train_slice]),
        f"{target_name}_forward_val_rmse": rmse(y_full[val_slice], y_pred[val_slice]),
        f"{target_name}_forward_test_rmse": rmse(y_full[test_slice], y_pred[test_slice]),
        f"{target_name}_forward_train_r2": r2(y_full[train_slice], y_pred[train_slice]),
        f"{target_name}_forward_val_r2": r2(y_full[val_slice], y_pred[val_slice]),
        f"{target_name}_forward_test_r2": r2(y_full[test_slice], y_pred[test_slice]),
        f"{target_name}_forward_all_mse": mse(y_full, y_pred),
        f"{target_name}_forward_all_rmse": rmse(y_full, y_pred),
        f"{target_name}_forward_all_r2": r2(y_full, y_pred),
    }

    if "time_s" in df.columns:
        time_axis = df["time_s"].to_numpy(dtype=float)
        x_label = "time_s"
    else:
        time_axis = np.arange(len(df), dtype=float)
        x_label = "index"

    pred_df = pd.DataFrame(
        {
            x_label: time_axis,
            f"{target_name}_true": y_full,
            f"{target_name}_pred": y_pred,
            f"{target_name}_residual": y_pred - y_full,
        }
    )
    pred_path = out_dir / f"sindy_trial2_forward_predictions_{target_name}.csv"
    pred_df.to_csv(pred_path, index=False)

    plot_path = out_dir / f"sindy_trial2_forward_overlay_{target_name}.png"
    plt.figure(figsize=(10, 5.5))
    plt.plot(time_axis, y_full, color="tab:blue", lw=2.0, label=f"{target_name} true")
    plt.plot(time_axis, y_pred, color="tab:red", lw=1.8, label=f"{target_name} predicted")
    plt.xlabel(x_label)
    plt.ylabel(target_name)
    plt.title(f"SINDy Trial 2 Forward Comparison - {target_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    forward_metrics[f"{target_name}_forward_predictions_file"] = pred_path.name
    forward_metrics[f"{target_name}_forward_plot_file"] = plot_path.name
    return forward_metrics


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    processed_dir = Path(__file__).resolve().parents[3] / "data" / "processed"
    df = load_data(processed_dir)
    validate_columns(df)

    n_before = len(df)
    # Removed zero-filtering: keep all data including hc=0 and he=0 values.

    n_after = len(df)
    if n_after == 0:
        raise ValueError("Dataset is empty.")

    train_slice, val_slice, test_slice = split_indices(n_after)
    n_train = train_slice.stop - train_slice.start
    n_val = val_slice.stop - val_slice.start
    n_test = test_slice.stop - test_slice.start

    print("Loaded dataset successfully.")
    print(f"Rows (before hc>0 filter): {n_before}")
    print(f"Rows (after  hc>0 filter): {n_after}")
    print(f"Sequential split -> train: {n_train}, val: {n_val}, test: {n_test}")
    print("Running STLSQ sparse regression Trial 2 for hc and he...")

    X_hc, hc_features = build_library_hc(df)
    y_hc = df["hc"].to_numpy(dtype=float)
    hc_summary, hc_coeffs = run_sparse_regression(
        name="hc",
        X=X_hc,
        y=y_hc,
        feature_names=hc_features,
        train_slice=train_slice,
        val_slice=val_slice,
        test_slice=test_slice,
        out_dir=base_dir,
    )

    X_he, he_features = build_library_he(df)
    y_he = df["he"].to_numpy(dtype=float)
    he_summary, he_coeffs = run_sparse_regression(
        name="he",
        X=X_he,
        y=y_he,
        feature_names=he_features,
        train_slice=train_slice,
        val_slice=val_slice,
        test_slice=test_slice,
        out_dir=base_dir,
    )

    hc_forward = evaluate_forward_predictions(
        target_name="hc",
        X_full=X_hc,
        y_full=y_hc,
        coeffs=hc_coeffs,
        df=df,
        train_slice=train_slice,
        val_slice=val_slice,
        test_slice=test_slice,
        out_dir=base_dir,
    )
    he_forward = evaluate_forward_predictions(
        target_name="he",
        X_full=X_he,
        y_full=y_he,
        coeffs=he_coeffs,
        df=df,
        train_slice=train_slice,
        val_slice=val_slice,
        test_slice=test_slice,
        out_dir=base_dir,
    )

    hc_summary.update(hc_forward)
    he_summary.update(he_forward)

    summary_df = pd.DataFrame([hc_summary, he_summary])
    summary_path = base_dir / "phase1_sindy_trial2_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("=" * 72)
    print("Sparse regression Trial 2 complete.")
    print(f"Summary saved to: {summary_path.name}")
    print(summary_df[["target", "selected_threshold", "n_nonzero", "val_mse", "val_r2", "test_mse", "test_r2"]])
    print("Forward-comparison metrics (all rows):")
    print(summary_df[["target", "hc_forward_all_rmse", "hc_forward_all_r2", "he_forward_all_rmse", "he_forward_all_r2"]].fillna("-"))


if __name__ == "__main__":
    main()
