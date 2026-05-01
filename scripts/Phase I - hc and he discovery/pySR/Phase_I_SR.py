"""
Phase I Symbolic Regression (PySR)
----------------------------------
Finalized workflow:
1) Run he discovery with sequential 70/20/10 split and test evaluation.
2) Run three hc approaches with the same split and test evaluation:
   - log(hc) target
   - hc^3 target
   - exact compound equation baseline (Dunkle form)


"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp

try:
    from pysr import PySRRegressor
except ImportError as exc:
    raise ImportError(
        "PySR is not installed. Install it with: pip install pysr"
    ) from exc

NITER = 250
NPOP = 16
POPSIZE = 40
TRAIN_FRAC = 0.70
VAL_FRAC = 0.20
LOG_EPS = 1e-12
B_DUNKLE = 268.9e3
A_DUNKLE = 0.884


def load_dunkle_clean(data_dir: Path) -> pd.DataFrame:
    file_path = data_dir / "dunkle_clean.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find file: {file_path.name}")
    return pd.read_csv(file_path)


def validate_columns(df: pd.DataFrame) -> None:
    required = {"Tw", "Tgi", "Pw", "Pgi", "hc", "he"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def real_cuberoot(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.abs(x) ** (1.0 / 3.0)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom <= 0.0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def split_indices(n: int) -> tuple[slice, slice, slice]:
    n_train = int(TRAIN_FRAC * n)
    n_val = int(VAL_FRAC * n)
    n_test = n - n_train - n_val
    train_slice = slice(0, n_train)
    val_slice = slice(n_train, n_train + n_val)
    test_slice = slice(n_train + n_val, n_train + n_val + n_test)
    return train_slice, val_slice, test_slice


def build_split_masks(n: int, train_slice: slice, val_slice: slice, test_slice: slice) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[train_slice] = True
    val_mask[val_slice] = True
    test_mask[test_slice] = True
    return train_mask, val_mask, test_mask


def plot_discovered_vs_actual(
    out_dir: Path,
    case_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    y_label: str,
) -> Path:
    idx = np.arange(len(y_true))
    plt.figure(figsize=(11, 5.5))
    plt.plot(idx, y_true, color="black", lw=1.8, label=f"Actual {y_label}")
    plt.plot(idx, y_pred, color="tab:orange", lw=1.6, ls="--", label=f"Discovered {y_label}")

    plt.scatter(idx[train_mask], y_true[train_mask], s=12, c="tab:blue", alpha=0.45, label="Train")
    plt.scatter(idx[val_mask], y_true[val_mask], s=12, c="tab:green", alpha=0.45, label="Val")
    plt.scatter(idx[test_mask], y_true[test_mask], s=12, c="tab:red", alpha=0.45, label="Test")

    plt.xlabel("Sequential sample index")
    plt.ylabel(y_label)
    plt.title(f"Discovered vs Actual ({case_name})")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    out_path = out_dir / f"overlay_discovered_vs_actual_{case_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def build_hc_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dT = (df["Tw"] - df["Tgi"]).to_numpy(dtype=float)
    dP = (df["Pw"] - df["Pgi"]).to_numpy(dtype=float)
    TwK = (df["Tw"] + 273.15).to_numpy(dtype=float)
    Pw = df["Pw"].to_numpy(dtype=float)
    compound_seed = dT + (dP * TwK) / (B_DUNKLE - Pw)

    X = np.column_stack([dT, dP, TwK, Pw, compound_seed])
    feature_names = ["dT", "dP", "TwK", "Pw", "compound_seed"]
    return X, feature_names, dT, dP, TwK, Pw


def build_he_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    dT = (df["Tw"] - df["Tgi"]).to_numpy(dtype=float)
    dP = (df["Pw"] - df["Pgi"]).to_numpy(dtype=float)
    hc = df["hc"].to_numpy(dtype=float)
    X = np.column_stack([dT, dP, hc])
    feature_names = ["dT", "dP", "hc"]
    return X, feature_names


def build_model(mode: str, random_state: int) -> PySRRegressor:
    common = dict(
        niterations=NITER,
        populations=NPOP,
        population_size=POPSIZE,
        maxsize=20,
        model_selection="best",
        progress=True,
        random_state=random_state,
        deterministic=True,
        parallelism="serial",
    )

    if mode == "he":
        return PySRRegressor(
            binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=[],
            elementwise_loss="loss(x, y) = (x - y)^2",
            **common,
        )

    if mode == "hc_log":
        return PySRRegressor(
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["safe_log(x::T) where {T} = x > T(0) ? log(x) : T(NaN)"],
            extra_sympy_mappings={"safe_log": lambda x: sp.log(x)},
            elementwise_loss="loss(x, y) = (x - y)^2",
            **common,
        )

    if mode == "hc_cube":
        return PySRRegressor(
            binary_operators=["+", "-", "*", "/"],
            unary_operators=[],
            elementwise_loss="loss(x, y) = (x - y)^2 + 1000 * max(-x, 0)^2",
            **common,
        )

    raise ValueError(f"Unsupported mode: {mode}")


def predict_hc_from_target(mode: str, y_pred: np.ndarray) -> np.ndarray:
    if mode == "hc_log":
        return np.exp(y_pred)
    if mode == "hc_cube":
        return real_cuberoot(y_pred)
    raise ValueError(f"Unsupported hc mode: {mode}")


def run_pysr_and_select(
    mode: str,
    X: np.ndarray,
    y_raw: np.ndarray,
    feature_names: list[str],
    train_slice: slice,
    val_slice: slice,
    test_slice: slice,
    random_state: int,
    out_dir: Path,
) -> tuple[dict[str, float], str, np.ndarray]:
    if mode == "hc_log":
        y = np.log(np.maximum(y_raw, LOG_EPS))
    elif mode == "hc_cube":
        y = y_raw**3
    else:
        y = y_raw

    X_train, X_val, X_test = X[train_slice], X[val_slice], X[test_slice]
    y_train, y_val, y_test = y[train_slice], y[val_slice], y[test_slice]
    y_raw_train, y_raw_val, y_raw_test = y_raw[train_slice], y_raw[val_slice], y_raw[test_slice]

    model = build_model(mode=mode, random_state=random_state)
    model.fit(X_train, y_train, variable_names=feature_names)
    eq = model.equations_.copy()

    val_mse = []
    train_mse = []
    test_mse = []
    val_mse_raw = []
    train_mse_raw = []
    test_mse_raw = []

    for i in range(len(eq)):
        try:
            y_pred_train = model.predict(X_train, index=i)
            y_pred_val = model.predict(X_val, index=i)
            y_pred_test = model.predict(X_test, index=i)

            train_mse.append(mse(y_train, y_pred_train))
            val_mse.append(mse(y_val, y_pred_val))
            test_mse.append(mse(y_test, y_pred_test))

            if mode in ("hc_log", "hc_cube"):
                y_pred_train_raw = predict_hc_from_target(mode, y_pred_train)
                y_pred_val_raw = predict_hc_from_target(mode, y_pred_val)
                y_pred_test_raw = predict_hc_from_target(mode, y_pred_test)
                train_mse_raw.append(mse(y_raw_train, y_pred_train_raw))
                val_mse_raw.append(mse(y_raw_val, y_pred_val_raw))
                test_mse_raw.append(mse(y_raw_test, y_pred_test_raw))
            else:
                train_mse_raw.append(np.nan)
                val_mse_raw.append(np.nan)
                test_mse_raw.append(np.nan)
        except Exception:
            train_mse.append(np.nan)
            val_mse.append(np.nan)
            test_mse.append(np.nan)
            train_mse_raw.append(np.nan)
            val_mse_raw.append(np.nan)
            test_mse_raw.append(np.nan)

    eq["train_mse"] = train_mse
    eq["val_mse"] = val_mse
    eq["test_mse"] = test_mse
    eq["train_mse_raw"] = train_mse_raw
    eq["val_mse_raw"] = val_mse_raw
    eq["test_mse_raw"] = test_mse_raw

    if mode in ("hc_log", "hc_cube"):
        eq["train_mse_forward"] = eq["train_mse_raw"]
        eq["val_mse_forward"] = eq["val_mse_raw"]
        eq["test_mse_forward"] = eq["test_mse_raw"]
    else:
        eq["train_mse_forward"] = eq["train_mse"]
        eq["val_mse_forward"] = eq["val_mse"]
        eq["test_mse_forward"] = eq["test_mse"]

    if mode in ("hc_log", "hc_cube"):
        sel_col = "val_mse_raw"
    else:
        sel_col = "val_mse"

    valid_idx = np.where(np.isfinite(eq[sel_col].to_numpy(dtype=float)))[0]
    if len(valid_idx) == 0:
        raise RuntimeError(f"No finite validation scores for mode {mode}.")
    best_idx = int(valid_idx[np.argmin(eq[sel_col].to_numpy(dtype=float)[valid_idx])])

    expr_col = "equation" if "equation" in eq.columns else None
    if expr_col is None:
        for candidate in ["sympy_format", "lambda_format", "jax_format"]:
            if candidate in eq.columns:
                expr_col = candidate
                break

    eq_path = out_dir / f"pysr_equations_{mode}.csv"
    eq.to_csv(eq_path, index=False)

    x_axis = np.arange(len(eq))
    finite = np.isfinite(eq["train_mse"].to_numpy(dtype=float)) & np.isfinite(eq["val_mse"].to_numpy(dtype=float))
    if np.any(finite):
        plt.figure(figsize=(9, 5))
        plt.plot(x_axis[finite], eq.loc[finite, "train_mse"].to_numpy(dtype=float), lw=1.6, label="train_mse")
        plt.plot(x_axis[finite], eq.loc[finite, "val_mse"].to_numpy(dtype=float), lw=1.6, label="val_mse")
        plt.xlabel("Search progression index")
        plt.ylabel("MSE (target space)")
        plt.title(f"MSE Progression Proxy - {mode}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"mse_progress_{mode}.png", dpi=150, bbox_inches="tight")
        plt.close()

    y_pred_train_best = model.predict(X_train, index=best_idx)
    y_pred_val_best = model.predict(X_val, index=best_idx)
    y_pred_test_best = model.predict(X_test, index=best_idx)

    summary = {
        "mode": mode,
        "train_mse_target": mse(y_train, y_pred_train_best),
        "val_mse_target": mse(y_val, y_pred_val_best),
        "test_mse_target": mse(y_test, y_pred_test_best),
        "train_rmse_target": rmse(y_train, y_pred_train_best),
        "val_rmse_target": rmse(y_val, y_pred_val_best),
        "test_rmse_target": rmse(y_test, y_pred_test_best),
        "train_r2_target": r2(y_train, y_pred_train_best),
        "val_r2_target": r2(y_val, y_pred_val_best),
        "test_r2_target": r2(y_test, y_pred_test_best),
        "selected_by": sel_col,
        "eq_file": str(eq_path.name),
    }

    y_pred_full_target = model.predict(X, index=best_idx)

    if mode in ("hc_log", "hc_cube"):
        y_pred_train_raw = predict_hc_from_target(mode, y_pred_train_best)
        y_pred_val_raw = predict_hc_from_target(mode, y_pred_val_best)
        y_pred_test_raw = predict_hc_from_target(mode, y_pred_test_best)
        y_pred_full_forward = predict_hc_from_target(mode, y_pred_full_target)
        summary.update(
            {
                "train_mse_forward": mse(y_raw_train, y_pred_train_raw),
                "val_mse_forward": mse(y_raw_val, y_pred_val_raw),
                "test_mse_forward": mse(y_raw_test, y_pred_test_raw),
                "train_mse_hc": mse(y_raw_train, y_pred_train_raw),
                "val_mse_hc": mse(y_raw_val, y_pred_val_raw),
                "test_mse_hc": mse(y_raw_test, y_pred_test_raw),
                "train_rmse_hc": rmse(y_raw_train, y_pred_train_raw),
                "val_rmse_hc": rmse(y_raw_val, y_pred_val_raw),
                "test_rmse_hc": rmse(y_raw_test, y_pred_test_raw),
                "train_r2_hc": r2(y_raw_train, y_pred_train_raw),
                "val_r2_hc": r2(y_raw_val, y_pred_val_raw),
                "test_r2_hc": r2(y_raw_test, y_pred_test_raw),
            }
        )
    else:
        y_pred_full_forward = y_pred_full_target
        summary.update(
            {
                "train_mse_forward": mse(y_train, y_pred_train_best),
                "val_mse_forward": mse(y_val, y_pred_val_best),
                "test_mse_forward": mse(y_test, y_pred_test_best),
            }
        )

    best_expr = str(eq.iloc[best_idx][expr_col]) if expr_col is not None else str(model.get_best(index=best_idx))
    return summary, best_expr, y_pred_full_forward


def run_exact_compound_baseline(
    dT: np.ndarray,
    dP: np.ndarray,
    TwK: np.ndarray,
    Pw: np.ndarray,
    hc_true: np.ndarray,
    train_slice: slice,
    val_slice: slice,
    test_slice: slice,
) -> tuple[dict[str, float], np.ndarray]:
    inner = dT + (dP * TwK) / (B_DUNKLE - Pw)
    hc_pred = A_DUNKLE * real_cuberoot(inner)

    y_train, y_val, y_test = hc_true[train_slice], hc_true[val_slice], hc_true[test_slice]
    p_train, p_val, p_test = hc_pred[train_slice], hc_pred[val_slice], hc_pred[test_slice]

    summary = {
        "mode": "hc_exact_compound",
        "train_mse_forward": mse(y_train, p_train),
        "val_mse_forward": mse(y_val, p_val),
        "test_mse_forward": mse(y_test, p_test),
        "train_mse_hc": mse(y_train, p_train),
        "val_mse_hc": mse(y_val, p_val),
        "test_mse_hc": mse(y_test, p_test),
        "train_rmse_hc": rmse(y_train, p_train),
        "val_rmse_hc": rmse(y_val, p_val),
        "test_rmse_hc": rmse(y_test, p_test),
        "train_r2_hc": r2(y_train, p_train),
        "val_r2_hc": r2(y_val, p_val),
        "test_r2_hc": r2(y_test, p_test),
        "selected_by": "fixed_equation",
        "eq_file": "n/a",
    }
    return summary, hc_pred


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    processed_dir = Path(__file__).resolve().parents[3] / "data" / "processed"
    df = load_dunkle_clean(processed_dir)
    validate_columns(df)

    n_before = len(df)
    n_after = len(df)
    if n_after == 0:
        raise ValueError("All rows were removed after filtering hc > 0.")

    print("Loaded dataset successfully.")

    train_slice, val_slice, test_slice = split_indices(n_after)
    n_train = train_slice.stop - train_slice.start
    n_val = val_slice.stop - val_slice.start
    n_test = test_slice.stop - test_slice.start
    print(f"Split sizes -> train: {n_train}, val: {n_val}, test: {n_test}")
    print(f"PySR budget -> niterations: {NITER}, populations: {NPOP}, population_size: {POPSIZE}")

    train_mask, val_mask, test_mask = build_split_masks(n_after, train_slice, val_slice, test_slice)

    summaries: list[dict[str, float]] = []

    # he run (with test-set evaluation)
    X_he, he_features = build_he_features(df)
    he_target = df["he"].to_numpy(dtype=float)
    he_summary, he_expr, he_pred = run_pysr_and_select(
        mode="he",
        X=X_he,
        y_raw=he_target,
        feature_names=he_features,
        train_slice=train_slice,
        val_slice=val_slice,
        test_slice=test_slice,
        random_state=43,
        out_dir=script_dir,
    )
    summaries.append(he_summary)
    print("=" * 72)
    print("Best he equation (selected by val_mse):")
    print(he_expr)
    print(f"he test RMSE: {he_summary['test_rmse_target']:.6e}")
    he_plot = plot_discovered_vs_actual(
        out_dir=script_dir,
        case_name="he",
        y_true=he_target,
        y_pred=he_pred,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        y_label="he",
    )
    print(f"Saved overlay plot: {he_plot.name}")

    # hc runs (two PySR approaches + exact-compound baseline)
    X_hc, hc_features, dT, dP, TwK, Pw = build_hc_features(df)
    hc_target = df["hc"].to_numpy(dtype=float)

    hc_log_summary, hc_log_expr, hc_log_pred = run_pysr_and_select(
        mode="hc_log",
        X=X_hc,
        y_raw=hc_target,
        feature_names=hc_features,
        train_slice=train_slice,
        val_slice=val_slice,
        test_slice=test_slice,
        random_state=42,
        out_dir=script_dir,
    )
    summaries.append(hc_log_summary)
    print("=" * 72)
    print("Best hc_log equation (selected by val_mse_raw):")
    print(hc_log_expr)
    print(f"hc_log test RMSE (hc space): {hc_log_summary['test_rmse_hc']:.6e}")
    hc_log_plot = plot_discovered_vs_actual(
        out_dir=script_dir,
        case_name="hc_log",
        y_true=hc_target,
        y_pred=hc_log_pred,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        y_label="hc",
    )
    print(f"Saved overlay plot: {hc_log_plot.name}")

    hc_cube_summary, hc_cube_expr, hc_cube_pred = run_pysr_and_select(
        mode="hc_cube",
        X=X_hc,
        y_raw=hc_target,
        feature_names=hc_features,
        train_slice=train_slice,
        val_slice=val_slice,
        test_slice=test_slice,
        random_state=44,
        out_dir=script_dir,
    )
    summaries.append(hc_cube_summary)
    print("=" * 72)
    print("Best hc_cube equation (selected by val_mse_raw):")
    print(hc_cube_expr)
    print(f"hc_cube test RMSE (hc space): {hc_cube_summary['test_rmse_hc']:.6e}")
    hc_cube_plot = plot_discovered_vs_actual(
        out_dir=script_dir,
        case_name="hc_cube",
        y_true=hc_target,
        y_pred=hc_cube_pred,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        y_label="hc",
    )
    print(f"Saved overlay plot: {hc_cube_plot.name}")

    hc_exact_summary, hc_exact_pred = run_exact_compound_baseline(
        dT=dT,
        dP=dP,
        TwK=TwK,
        Pw=Pw,
        hc_true=hc_target,
        train_slice=train_slice,
        val_slice=val_slice,
        test_slice=test_slice,
    )
    summaries.append(hc_exact_summary)
    print("=" * 72)
    print("Exact compound baseline:")
    print(f"hc_exact test RMSE (hc space): {hc_exact_summary['test_rmse_hc']:.6e}")
    hc_exact_plot = plot_discovered_vs_actual(
        out_dir=script_dir,
        case_name="hc_exact_compound",
        y_true=hc_target,
        y_pred=hc_exact_pred,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        y_label="hc",
    )
    print(f"Saved overlay plot: {hc_exact_plot.name}")

    summary_df = pd.DataFrame(summaries)
    summary_path = script_dir / "phase1_summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    print("=" * 72)
    print(f"Summary metrics saved to: {summary_path}")
    print(summary_df[["mode", "selected_by", "val_mse_hc", "test_mse_hc", "test_rmse_hc"]].fillna(np.nan))


if __name__ == "__main__":
    main()