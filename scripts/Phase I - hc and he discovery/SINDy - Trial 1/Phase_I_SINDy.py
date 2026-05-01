"""
Phase I - Sparse Regression (STLSQ) for Dunkle model rediscovery

This script implements the second approach of Phase I:
- Target equations: hc and he
- Candidate libraries:
  * hc: Tw, Tgi, Pw, Pgi, dT, dP, dP/dT, dT/dP,
        compound_cuberoot = cbrt(dT + (dP * TwK) / (B - Pw)),
        and all polynomial combinations up to second order.
  * he: Tw, Tgi, Pw, Pgi, dT, hc, dP, dP/dT, dT/dP,
        and all polynomial combinations up to second order.
- Split: sequential 70% train / 20% val / 10% test
- Sparsity: Sequential Thresholded Least Squares (STLSQ)
- Threshold selection: validation-driven with sparsity preference

Outputs:
- sindy_selection_hc.csv
- sindy_selection_he.csv
- sindy_coefficients_hc.csv
- sindy_coefficients_he.csv
- sindy_loss_progress_hc.png
- sindy_loss_progress_he.png
- phase1_sindy_summary.csv
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TRAIN_FRAC = 0.70
VAL_FRAC = 0.20
B_DUNKLE = 268.9e3
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


def real_cuberoot(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.abs(x) ** (1.0 / 3.0)


def add_second_order_terms(base: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    terms = dict(base)
    names = list(base.keys())
    for a, b in combinations_with_replacement(names, 2):
        terms[f"{a}*{b}"] = base[a] * base[b]
    return terms


def build_library_hc(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    Tw = df["Tw"].to_numpy(dtype=float)
    Tgi = df["Tgi"].to_numpy(dtype=float)
    Pw = df["Pw"].to_numpy(dtype=float)
    Pgi = df["Pgi"].to_numpy(dtype=float)

    dT = Tw - Tgi
    dP = Pw - Pgi
    TwK = Tw + 273.15

    base = {
        "Tw": Tw,
        "Tgi": Tgi,
        "Pw": Pw,
        "Pgi": Pgi,
        "dT": dT,
        "dP": dP,
        "dP_over_dT": safe_divide(dP, dT),
        "dT_over_dP": safe_divide(dT, dP),
        "compound_cuberoot": real_cuberoot(dT + (dP * TwK) / (B_DUNKLE - Pw)),
    }

    terms = add_second_order_terms(base)
    feature_names = ["1"] + list(terms.keys())
    X = np.column_stack([np.ones(len(df), dtype=float)] + [terms[k] for k in terms])
    return X, feature_names


def build_library_he(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    Tw = df["Tw"].to_numpy(dtype=float)
    Tgi = df["Tgi"].to_numpy(dtype=float)
    Pw = df["Pw"].to_numpy(dtype=float)
    Pgi = df["Pgi"].to_numpy(dtype=float)
    hc = df["hc"].to_numpy(dtype=float)

    dT = Tw - Tgi
    dP = Pw - Pgi

    base = {
        "Tw": Tw,
        "Tgi": Tgi,
        "Pw": Pw,
        "Pgi": Pgi,
        "dT": dT,
        "hc": hc,
        "dP": dP,
        "dP_over_dT": safe_divide(dP, dT),
        "dT_over_dP": safe_divide(dT, dP),
    }

    terms = add_second_order_terms(base)
    feature_names = ["1"] + list(terms.keys())
    X = np.column_stack([np.ones(len(df), dtype=float)] + [terms[k] for k in terms])
    return X, feature_names


def solve_ridge(X: np.ndarray, y: np.ndarray, ridge: float = RIDGE) -> np.ndarray:
    xtx = X.T @ X
    rhs = X.T @ y
    eye = np.eye(xtx.shape[0])

    # Adaptive Tikhonov regularization to avoid singular solves.
    lam = max(float(ridge), 1e-10)
    for _ in range(8):
        try:
            return np.linalg.solve(xtx + lam * eye, rhs)
        except np.linalg.LinAlgError:
            lam *= 10.0

    # Final fallback for severely ill-conditioned systems.
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
    # Balance fit and sparsity: keep models within 2% of best val MSE, choose sparsest among them.
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
) -> dict[str, float]:
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

    # Build coefficient table for selected model.
    coeff_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coeffs_best,
            "abs_coefficient": np.abs(coeffs_best),
            "is_active": np.abs(coeffs_best) > 0.0,
        }
    ).sort_values(["is_active", "abs_coefficient"], ascending=[False, False])

    selection_df = pd.DataFrame(records)

    coeff_path = out_dir / f"sindy_coefficients_{name}.csv"
    selection_path = out_dir / f"sindy_selection_{name}.csv"
    coeff_df.to_csv(coeff_path, index=False)
    selection_df.to_csv(selection_path, index=False)

    # Plot train/validation loss progression across threshold sweep order.
    plot_path = out_dir / f"sindy_loss_progress_{name}.png"
    x = np.arange(len(selection_df))
    plt.figure(figsize=(9, 5))
    plt.plot(x, selection_df["train_mse"].to_numpy(dtype=float), lw=1.8, label="train_mse")
    plt.plot(x, selection_df["val_mse"].to_numpy(dtype=float), lw=1.8, label="val_mse")
    plt.xlabel("Threshold sweep index")
    plt.ylabel("MSE")
    plt.title(f"STLSQ Train/Validation Loss Progression - {name}")
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
    return summary


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    processed_dir = Path(__file__).resolve().parents[3] / "data" / "processed"
    df = load_data(processed_dir)
    validate_columns(df)

    n_before = len(df)
    df = df[df["hc"] > 0.0].reset_index(drop=True)
    n_after = len(df)
    if n_after == 0:
        raise ValueError("All rows removed after filtering hc > 0.")

    train_slice, val_slice, test_slice = split_indices(n_after)
    n_train = train_slice.stop - train_slice.start
    n_val = val_slice.stop - val_slice.start
    n_test = test_slice.stop - test_slice.start

    print("Loaded dataset successfully.")
    print(f"Rows (before hc>0 filter): {n_before}")
    print(f"Rows (after  hc>0 filter): {n_after}")
    print(f"Sequential split -> train: {n_train}, val: {n_val}, test(held-out): {n_test}")
    print("Running STLSQ sparse regression for hc and he...")

    X_hc, hc_features = build_library_hc(df)
    y_hc = df["hc"].to_numpy(dtype=float)
    hc_summary = run_sparse_regression(
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
    he_summary = run_sparse_regression(
        name="he",
        X=X_he,
        y=y_he,
        feature_names=he_features,
        train_slice=train_slice,
        val_slice=val_slice,
        test_slice=test_slice,
        out_dir=base_dir,
    )

    summary_df = pd.DataFrame([hc_summary, he_summary])
    summary_path = base_dir / "phase1_sindy_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("=" * 72)
    print("Sparse regression complete.")
    print(f"Summary saved to: {summary_path.name}")
    print(summary_df[["target", "selected_threshold", "n_nonzero", "val_mse", "val_r2", "test_mse", "test_r2"]])


if __name__ == "__main__":
    main()

