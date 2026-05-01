"""
Phase II - PINN-style parameter discovery using analytical ODE update

Target recurrence:
    Tw(t+dt) = F(t)/a + (Tw(t) - F(t)/a) * exp(-a * dt)

With trainable parameters:
    a = c1
    F(t) = c2*Ta(t) + c3*Tgi(t) + c4*I(t) + c5

Residual form on measured data:
    Tw_meas(t+dt) - [F(t)/a + (Tw_meas(t) - F(t)/a) * exp(-a*dt)] = 0

Training objective:
- Data loss: predicted Tw rollout vs measured Tw (train window)
- Residual loss: recurrence residual on measured Tw pairs (train rows)
- IC loss: (Tw_pred(0) - 23)^2

Split:
- Sequential 70% train, 20% val, 10% test (test evaluated)

Outputs:
- phase2_pinn_selection_dTw_dt.csv
- phase2_pinn_coefficients_dTw_dt.csv
- phase2_pinn_summary.csv
- phase2_pinn_rollout_Tw.csv
- phase2_pinn_rollout_Tw_plot.png
- phase2_pinn_loss_curves.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


TRAIN_FRAC = 0.70
VAL_FRAC = 0.20
EPS = 1e-12
SEED = 42
IC_TW0 = 23.0
W_DATA = 1.0
W_RES = 1.0
W_IC = 10.0
PRINT_EPOCH_DIAGNOSTICS = False


def load_data(base_dir: Path) -> pd.DataFrame:
    candidate_paths = [base_dir / "dunkle_clean.csv", base_dir.parent / "dunkle_clean.csv"]
    for path in candidate_paths:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError("Missing file: dunkle_clean.csv")


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


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom <= 0.0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class AnalyticalPINN(nn.Module):
    """
    Trainable constants in analytical recurrence:
        Tw_{k+1} = F_k/a + (Tw_k - F_k/a) * exp(-a * dt_k)
        F_k = c2*Ta_k + c3*Tgi_k + c4*I_k + c5
        a = c1
    """

    def __init__(self, c1_init: float, c2_init: float, c3_init: float, c4_init: float, c5_init: float, tw0_init: float) -> None:
        super().__init__()
        self.c1_raw = nn.Parameter(torch.tensor(float(c1_init), dtype=torch.float64))
        self.c2 = nn.Parameter(torch.tensor(float(c2_init), dtype=torch.float64))
        self.c3 = nn.Parameter(torch.tensor(float(c3_init), dtype=torch.float64))
        self.c4 = nn.Parameter(torch.tensor(float(c4_init), dtype=torch.float64))
        self.c5 = nn.Parameter(torch.tensor(float(c5_init), dtype=torch.float64))
        self.tw0 = nn.Parameter(torch.tensor(float(tw0_init), dtype=torch.float64))

    def effective_c1(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.c1_raw) + EPS

    def forcing(self, Ta: torch.Tensor, Tgi: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
        return self.c2 * Ta + self.c3 * Tgi + self.c4 * I + self.c5

    def rollout(self, dt: torch.Tensor, Ta: torch.Tensor, Tgi: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
        tw_vals = [self.tw0]
        c1 = self.effective_c1()
        F = self.forcing(Ta, Tgi, I)
        eq = F / c1
        for k in range(int(dt.shape[0])):
            tw_next = eq[k] + (tw_vals[k] - eq[k]) * torch.exp(-c1 * dt[k])
            tw_vals.append(tw_next)
        return torch.stack(tw_vals)

    def residual_on_measured(self, Tw_meas: torch.Tensor, dt: torch.Tensor, Ta: torch.Tensor, Tgi: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
        c1 = self.effective_c1()
        F = self.forcing(Ta, Tgi, I)
        eq = F / c1
        tw_next_model = eq + (Tw_meas[:-1] - eq) * torch.exp(-c1 * dt)
        return Tw_meas[1:] - tw_next_model


def train_one_model(
    config: dict[str, float],
    t_full: np.ndarray,
    Tw_full: np.ndarray,
    Ta_full: np.ndarray,
    Tgi_full: np.ndarray,
    I_full: np.ndarray,
    train_slice: slice,
    val_slice: slice,
    test_slice: slice,
) -> tuple[AnalyticalPINN, dict[str, float], dict[str, list[float]]]:
    set_seed(SEED)

    model = AnalyticalPINN(
        c1_init=float(config["c1_init"]),
        c2_init=float(config["c2_init"]),
        c3_init=float(config["c3_init"]),
        c4_init=float(config["c4_init"]),
        c5_init=float(config["c5_init"]),
        tw0_init=float(config["tw0_init"]),
    ).double()

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    epochs = int(config["epochs"])

    dt_full = np.diff(t_full)
    if np.any(dt_full <= 0.0):
        raise ValueError("time_s must be strictly increasing.")

    Tw_train = Tw_full[: train_slice.stop + 1]
    dt_train = dt_full[: train_slice.stop]
    Ta_train = Ta_full[: train_slice.stop]
    Tgi_train = Tgi_full[: train_slice.stop]
    I_train = I_full[: train_slice.stop]

    Tw_train_t = torch.tensor(Tw_train, dtype=torch.float64)
    dt_train_t = torch.tensor(dt_train, dtype=torch.float64)
    Ta_train_t = torch.tensor(Ta_train, dtype=torch.float64)
    Tgi_train_t = torch.tensor(Tgi_train, dtype=torch.float64)
    I_train_t = torch.tensor(I_train, dtype=torch.float64)

    Tw_full_t = torch.tensor(Tw_full, dtype=torch.float64)
    dt_full_t = torch.tensor(dt_full, dtype=torch.float64)
    Ta_full_t = torch.tensor(Ta_full[:-1], dtype=torch.float64)
    Tgi_full_t = torch.tensor(Tgi_full[:-1], dtype=torch.float64)
    I_full_t = torch.tensor(I_full[:-1], dtype=torch.float64)

    history = {
        "train_total": [],
        "val_total": [],
        "train_data": [],
        "val_data": [],
        "train_res": [],
        "val_res": [],
    }

    for epoch_idx in range(epochs):
        optimizer.zero_grad()

        Tw_pred_train = model.rollout(dt_train_t, Ta_train_t, Tgi_train_t, I_train_t)
        loss_data = torch.mean((Tw_pred_train - Tw_train_t) ** 2)

        resid_train = model.residual_on_measured(Tw_train_t, dt_train_t, Ta_train_t, Tgi_train_t, I_train_t)
        loss_res = torch.mean(resid_train**2)

        loss_ic = (Tw_pred_train[0] - IC_TW0) ** 2

        loss = W_DATA * loss_data + W_RES * loss_res + W_IC * loss_ic
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            Tw_pred_full_epoch = model.rollout(dt_full_t, Ta_full_t, Tgi_full_t, I_full_t)
            train_data_epoch = torch.mean((Tw_pred_full_epoch[: train_slice.stop + 1] - Tw_full_t[: train_slice.stop + 1]) ** 2)
            val_data_epoch = torch.mean(
                (Tw_pred_full_epoch[train_slice.stop : val_slice.stop + 1] - Tw_full_t[train_slice.stop : val_slice.stop + 1]) ** 2
            )

            resid_full_epoch = model.residual_on_measured(Tw_full_t, dt_full_t, Ta_full_t, Tgi_full_t, I_full_t)
            train_res_epoch = torch.mean(resid_full_epoch[train_slice] ** 2)
            if val_slice.stop > val_slice.start:
                val_res_epoch = torch.mean(resid_full_epoch[val_slice] ** 2)
            else:
                val_res_epoch = torch.tensor(float("nan"), dtype=torch.float64)

            ic_epoch = (Tw_pred_full_epoch[0] - IC_TW0) ** 2
            train_total_epoch = W_DATA * train_data_epoch + W_RES * train_res_epoch + W_IC * ic_epoch
            val_total_epoch = W_DATA * val_data_epoch + W_RES * val_res_epoch + W_IC * ic_epoch

            history["train_total"].append(float(train_total_epoch.item()))
            history["val_total"].append(float(val_total_epoch.item()))
            history["train_data"].append(float(train_data_epoch.item()))
            history["val_data"].append(float(val_data_epoch.item()))
            history["train_res"].append(float(train_res_epoch.item()))
            history["val_res"].append(float(val_res_epoch.item()) if torch.isfinite(val_res_epoch) else float("nan"))

        if PRINT_EPOCH_DIAGNOSTICS:
            print(
                f"[diag] epoch={epoch_idx + 1:04d}/{epochs} "
                f"loss={float(loss.item()):.6e} "
                f"data={float(loss_data.item()):.6e} "
                f"res={float(loss_res.item()):.6e} "
                f"ic={float(loss_ic.item()):.6e}"
            )

    Tw_pred_full = rollout_numpy(model, t_full, Ta_full, Tgi_full, I_full)
    dTw_true = np.diff(Tw_full) / np.maximum(np.diff(t_full), EPS)
    dTw_pred = np.diff(Tw_pred_full) / np.maximum(np.diff(t_full), EPS)

    train_data_mse = mse(Tw_full[: train_slice.stop + 1], Tw_pred_full[: train_slice.stop + 1])
    val_data_mse = mse(Tw_full[train_slice.stop : val_slice.stop + 1], Tw_pred_full[train_slice.stop : val_slice.stop + 1])
    test_data_mse = mse(Tw_full[val_slice.stop : test_slice.stop + 1], Tw_pred_full[val_slice.stop : test_slice.stop + 1])

    train_rollout_deriv_mse = mse(dTw_true[train_slice], dTw_pred[train_slice])
    val_rollout_deriv_mse = mse(dTw_true[val_slice], dTw_pred[val_slice]) if val_slice.stop > val_slice.start else float("nan")
    test_rollout_deriv_mse = mse(dTw_true[test_slice], dTw_pred[test_slice]) if test_slice.stop > test_slice.start else float("nan")

    resid_train_np = residual_numpy_on_measured(model, Tw_train, dt_train, Ta_train, Tgi_train, I_train)
    train_residual_mse = float(np.mean(resid_train_np**2))

    Tw_val_meas = Tw_full[train_slice.stop : val_slice.stop + 1]
    t_val_meas = t_full[train_slice.stop : val_slice.stop + 1]
    dt_val = np.diff(t_val_meas)
    Ta_val = Ta_full[train_slice.stop : val_slice.stop]
    Tgi_val = Tgi_full[train_slice.stop : val_slice.stop]
    I_val = I_full[train_slice.stop : val_slice.stop]
    if len(Tw_val_meas) > 1:
        resid_val_np = residual_numpy_on_measured(model, Tw_val_meas, dt_val, Ta_val, Tgi_val, I_val)
        val_residual_mse = float(np.mean(resid_val_np**2))
    else:
        val_residual_mse = float("nan")

    Tw_test_meas = Tw_full[val_slice.stop : test_slice.stop + 1]
    t_test_meas = t_full[val_slice.stop : test_slice.stop + 1]
    dt_test = np.diff(t_test_meas)
    Ta_test = Ta_full[val_slice.stop : test_slice.stop]
    Tgi_test = Tgi_full[val_slice.stop : test_slice.stop]
    I_test = I_full[val_slice.stop : test_slice.stop]
    if len(Tw_test_meas) > 1:
        resid_test_np = residual_numpy_on_measured(model, Tw_test_meas, dt_test, Ta_test, Tgi_test, I_test)
        test_residual_mse = float(np.mean(resid_test_np**2))
    else:
        test_residual_mse = float("nan")

    results = {
        "train_data_mse": train_data_mse,
        "val_data_mse": val_data_mse,
        "test_data_mse": test_data_mse,
        "train_residual_mse": train_residual_mse,
        "val_residual_mse": val_residual_mse,
        "test_residual_mse": test_residual_mse,
        "train_rollout_deriv_mse": train_rollout_deriv_mse,
        "val_rollout_deriv_mse": val_rollout_deriv_mse,
        "test_rollout_deriv_mse": test_rollout_deriv_mse,
        "val_objective": float(val_rollout_deriv_mse),
    }
    return model, results, history


def residual_numpy_on_measured(
    model: AnalyticalPINN,
    Tw_meas: np.ndarray,
    dt: np.ndarray,
    Ta: np.ndarray,
    Tgi: np.ndarray,
    I: np.ndarray,
) -> np.ndarray:
    if len(Tw_meas) < 2:
        return np.array([], dtype=float)
    c1 = float(model.effective_c1().detach().cpu().item())
    c2 = float(model.c2.detach().cpu().item())
    c3 = float(model.c3.detach().cpu().item())
    c4 = float(model.c4.detach().cpu().item())
    c5 = float(model.c5.detach().cpu().item())
    F = c2 * Ta + c3 * Tgi + c4 * I + c5
    eq = F / max(c1, EPS)
    tw_next_model = eq + (Tw_meas[:-1] - eq) * np.exp(-c1 * dt)
    return Tw_meas[1:] - tw_next_model


def rollout_numpy(model: AnalyticalPINN, t: np.ndarray, Ta: np.ndarray, Tgi: np.ndarray, I: np.ndarray) -> np.ndarray:
    dt = np.diff(t)
    if np.any(dt <= 0.0):
        raise ValueError("time_s must be strictly increasing for rollout.")

    c1 = float(model.effective_c1().detach().cpu().item())
    c2 = float(model.c2.detach().cpu().item())
    c3 = float(model.c3.detach().cpu().item())
    c4 = float(model.c4.detach().cpu().item())
    c5 = float(model.c5.detach().cpu().item())
    tw0 = float(model.tw0.detach().cpu().item())

    F = c2 * Ta[:-1] + c3 * Tgi[:-1] + c4 * I[:-1] + c5
    eq = F / max(c1, EPS)

    tw_pred = np.zeros_like(t, dtype=float)
    tw_pred[0] = tw0
    for k in range(len(dt)):
        tw_pred[k + 1] = eq[k] + (tw_pred[k] - eq[k]) * np.exp(-c1 * dt[k])
    return tw_pred


def format_equation(c1: float, c2: float, c3: float, c4: float, c5: float) -> str:
    return (
        "Tw_next = (F/c1) + (Tw - F/c1)*exp(-c1*dt), "
        f"F = ({c2:.8e})*Ta + ({c3:.8e})*Tgi + ({c4:.8e})*I + ({c5:.8e}), "
        f"c1 = {c1:.8e}"
    )


def main() -> None:
    set_seed(SEED)
    base_dir = Path(__file__).resolve().parent
    processed_dir = Path(__file__).resolve().parents[3] / "data" / "processed"
    df = load_data(processed_dir)

    t_full = get_column(df, "time_s")
    I_full = get_column(df, "I")
    Ta_full = get_column(df, "Ta", fallback="T_a")
    Tgi_full = get_column(df, "Tgi")
    Tw_full = get_column(df, "Tw")

    if len(t_full) < 3:
        raise ValueError("Need at least 3 time points.")
    if np.any(np.diff(t_full) <= 0.0):
        raise ValueError("time_s must be strictly increasing.")

    y_true = np.diff(Tw_full) / np.maximum(np.diff(t_full), EPS)
    train_slice, val_slice, test_slice = split_indices(len(y_true))

    tw_mean = float(np.mean(Tw_full[: train_slice.stop + 1]))
    configs: list[dict[str, float]] = []
    for config_id, (lr, epochs) in enumerate(
        [
            (1e-2, 3000),
            (5e-3, 4000),
            (1e-3, 6000),
        ],
        start=1,
    ):
        configs.append(
            {
                "config_id": float(config_id),
                "lr": float(lr),
                "epochs": float(epochs),
                "c1_init": 1e-3,
                "c2_init": 0.0,
                "c3_init": 0.0,
                "c4_init": 0.0,
                "c5_init": tw_mean * 1e-3,
                "tw0_init": IC_TW0,
                "w_data": W_DATA,
                "w_res": W_RES,
                "w_ic": W_IC,
            }
        )

    records: list[dict[str, float]] = []
    models: list[AnalyticalPINN] = []
    histories: list[dict[str, list[float]]] = []
    for cfg in configs:
        model, metrics, history = train_one_model(
            config=cfg,
            t_full=t_full,
            Tw_full=Tw_full,
            Ta_full=Ta_full,
            Tgi_full=Tgi_full,
            I_full=I_full,
            train_slice=slice(0, train_slice.stop),
            val_slice=slice(train_slice.stop, val_slice.stop),
            test_slice=slice(val_slice.stop, test_slice.stop),
        )
        models.append(model)
        histories.append(history)
        records.append({**cfg, **metrics})

    val_objectives = np.array([r["val_objective"] for r in records], dtype=float)
    best_idx = int(np.nanargmin(val_objectives))
    best_rec = records[best_idx]
    best_model = models[best_idx]
    best_history = histories[best_idx]

    c1_best = float(best_model.effective_c1().detach().cpu().item())
    c2_best = float(best_model.c2.detach().cpu().item())
    c3_best = float(best_model.c3.detach().cpu().item())
    c4_best = float(best_model.c4.detach().cpu().item())
    c5_best = float(best_model.c5.detach().cpu().item())
    tw0_best = float(best_model.tw0.detach().cpu().item())

    Tw_pred_full = rollout_numpy(best_model, t_full, Ta_full, Tgi_full, I_full)
    dTw_pred = np.diff(Tw_pred_full) / np.maximum(np.diff(t_full), EPS)

    dTw_train_true = y_true[train_slice]
    dTw_val_true = y_true[val_slice]
    dTw_test_true = y_true[test_slice]
    dTw_train_pred = dTw_pred[train_slice]
    dTw_val_pred = dTw_pred[val_slice]
    dTw_test_pred = dTw_pred[test_slice]

    train_mse = mse(dTw_train_true, dTw_train_pred)
    val_mse = mse(dTw_val_true, dTw_val_pred) if len(dTw_val_true) > 0 else float("nan")
    test_mse = mse(dTw_test_true, dTw_test_pred) if len(dTw_test_true) > 0 else float("nan")
    train_rmse = rmse(dTw_train_true, dTw_train_pred)
    val_rmse = rmse(dTw_val_true, dTw_val_pred) if len(dTw_val_true) > 0 else float("nan")
    test_rmse = rmse(dTw_test_true, dTw_test_pred) if len(dTw_test_true) > 0 else float("nan")
    train_r2 = r2(dTw_train_true, dTw_train_pred)
    val_r2 = r2(dTw_val_true, dTw_val_pred) if len(dTw_val_true) > 0 else float("nan")
    test_r2 = r2(dTw_test_true, dTw_test_pred) if len(dTw_test_true) > 0 else float("nan")

    tw_train_slice = slice(0, train_slice.stop + 1)
    tw_val_slice = slice(train_slice.stop, val_slice.stop + 1)
    tw_test_slice = slice(val_slice.stop, test_slice.stop + 1)

    rollout_train_mse = mse(Tw_full[tw_train_slice], Tw_pred_full[tw_train_slice])
    rollout_val_mse = mse(Tw_full[tw_val_slice], Tw_pred_full[tw_val_slice]) if tw_val_slice.stop > tw_val_slice.start else float("nan")
    rollout_test_mse = mse(Tw_full[tw_test_slice], Tw_pred_full[tw_test_slice]) if tw_test_slice.stop > tw_test_slice.start else float("nan")
    rollout_train_rmse = rmse(Tw_full[tw_train_slice], Tw_pred_full[tw_train_slice])
    rollout_val_rmse = rmse(Tw_full[tw_val_slice], Tw_pred_full[tw_val_slice]) if tw_val_slice.stop > tw_val_slice.start else float("nan")
    rollout_test_rmse = rmse(Tw_full[tw_test_slice], Tw_pred_full[tw_test_slice]) if tw_test_slice.stop > tw_test_slice.start else float("nan")
    rollout_train_r2 = r2(Tw_full[tw_train_slice], Tw_pred_full[tw_train_slice])
    rollout_val_r2 = r2(Tw_full[tw_val_slice], Tw_pred_full[tw_val_slice]) if tw_val_slice.stop > tw_val_slice.start else float("nan")
    rollout_test_r2 = r2(Tw_full[tw_test_slice], Tw_pred_full[tw_test_slice]) if tw_test_slice.stop > tw_test_slice.start else float("nan")

    selection_df = pd.DataFrame(records)
    selection_path = base_dir / "phase2_pinn_selection_dTw_dt.csv"
    selection_df.to_csv(selection_path, index=False)

    coeff_df = pd.DataFrame(
        {
            "parameter": ["c1", "c2", "c3", "c4", "c5", "tw0"],
            "value": [c1_best, c2_best, c3_best, c4_best, c5_best, tw0_best],
            "description": [
                "a = c1 (decay rate)",
                "coefficient of Ta(t) in F(t)",
                "coefficient of Tgi(t) in F(t)",
                "coefficient of I(t) in F(t)",
                "bias term in F(t)",
                "learned initial condition for rollout",
            ],
        }
    )
    coeff_path = base_dir / "phase2_pinn_coefficients_dTw_dt.csv"
    coeff_df.to_csv(coeff_path, index=False)

    rollout_df = pd.DataFrame(
        {
            "time_s": t_full,
            "Tw_true": Tw_full,
            "Tw_pred_rollout": Tw_pred_full,
            "abs_error": np.abs(Tw_pred_full - Tw_full),
            "split": np.where(
                np.arange(len(t_full)) <= train_slice.stop,
                "train",
                np.where(np.arange(len(t_full)) <= val_slice.stop, "val", "test"),
            ),
        }
    )
    rollout_path = base_dir / "phase2_pinn_rollout_Tw.csv"
    rollout_df.to_csv(rollout_path, index=False)

    plt.figure(figsize=(10, 5.5))
    plt.plot(t_full, Tw_full, color="tab:blue", lw=2.0, label="Tw ground")
    plt.plot(t_full, Tw_pred_full, color="tab:red", lw=2.0, label="Tw predicted rollout")
    if 0 < train_slice.stop < len(t_full):
        plt.axvline(t_full[train_slice.stop], color="gray", lw=1.2, linestyle="--", label="train/val split")
    if 0 < val_slice.stop < len(t_full):
        plt.axvline(t_full[val_slice.stop], color="black", lw=1.2, linestyle=":", label="val/test split")
    plt.xlabel("time_s")
    plt.ylabel("Tw (degC)")
    plt.title("Phase II PINN Analytical Rollout: Tw Predicted vs Ground")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    rollout_plot_path = base_dir / "phase2_pinn_rollout_Tw_plot.png"
    plt.savefig(rollout_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    epochs_axis = np.arange(1, len(best_history["train_total"]) + 1)
    plt.figure(figsize=(10, 5.5))
    plt.plot(epochs_axis, best_history["train_total"], color="tab:blue", lw=2.0, label="Train total loss")
    plt.plot(epochs_axis, best_history["val_total"], color="tab:orange", lw=2.0, label="Validation total loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Phase II PINN Loss Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    loss_plot_path = base_dir / "phase2_pinn_loss_curves.png"
    plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    summary = {
        "target": "Tw analytical recurrence with F(t)",
        "test_evaluated": True,
        "selected_config_id": int(best_rec["config_id"]),
        "learning_rate": float(best_rec["lr"]),
        "epochs": int(best_rec["epochs"]),
        "w_data": W_DATA,
        "w_residual": W_RES,
        "w_ic": W_IC,
        "residual_points_train": int(train_slice.stop - train_slice.start),
        "train_mse": train_mse,
        "val_mse": val_mse,
        "test_mse": test_mse,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "val_r2": val_r2,
        "test_r2": test_r2,
        "rows_total_ode": int(len(y_true)),
        "train_points": int(train_slice.stop - train_slice.start),
        "val_points": int(val_slice.stop - val_slice.start),
        "test_points": int(test_slice.stop - test_slice.start),
        "rollout_tw0_target": IC_TW0,
        "rollout_tw0_learned": tw0_best,
        "rollout_train_mse": rollout_train_mse,
        "rollout_val_mse": rollout_val_mse,
        "rollout_test_mse": rollout_test_mse,
        "rollout_train_rmse": rollout_train_rmse,
        "rollout_val_rmse": rollout_val_rmse,
        "rollout_test_rmse": rollout_test_rmse,
        "rollout_train_r2": rollout_train_r2,
        "rollout_val_r2": rollout_val_r2,
        "rollout_test_r2": rollout_test_r2,
        "c1": c1_best,
        "c2": c2_best,
        "c3": c3_best,
        "c4": c4_best,
        "c5": c5_best,
        "equation": format_equation(c1_best, c2_best, c3_best, c4_best, c5_best),
        "selection_file": selection_path.name,
        "coefficients_file": coeff_path.name,
        "rollout_file": rollout_path.name,
        "rollout_plot_file": rollout_plot_path.name,
        "loss_plot_file": loss_plot_path.name,
    }
    summary_path = base_dir / "phase2_pinn_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print("=" * 72)
    print("Phase II PINN complete (analytical recurrence residual with F(t))")
    print("Test metrics evaluated and exported.")
    print(f"Selected config id: {int(best_rec['config_id'])}")
    print(f"Selected hyperparameters: lr={best_rec['lr']:.2e}, epochs={int(best_rec['epochs'])}")
    print(f"Loss weights: w_data={W_DATA:.2f}, w_res={W_RES:.2f}, w_ic={W_IC:.2f}")
    print(
        "Discovered parameters: "
        f"c1={c1_best:.8e}, c2={c2_best:.8e}, c3={c3_best:.8e}, c4={c4_best:.8e}, c5={c5_best:.8e}, tw0={tw0_best:.8e}"
    )
    print(f"Residual points (train): {int(train_slice.stop - train_slice.start)}")
    print(f"Validation MSE (dTw/dt from rollout): {val_mse:.6e}")
    print(f"Test MSE (dTw/dt from rollout): {test_mse:.6e}")
    print(f"Rollout validation MSE (Tw): {rollout_val_mse:.6e}")
    print(f"Rollout test MSE (Tw): {rollout_test_mse:.6e}")
    print(f"Selection saved: {selection_path.name}")
    print(f"Coefficients saved: {coeff_path.name}")
    print(f"Rollout saved: {rollout_path.name}")
    print(f"Rollout plot saved: {rollout_plot_path.name}")
    print(f"Loss plot saved: {loss_plot_path.name}")
    print(f"Summary saved: {summary_path.name}")
    print("=" * 72)


if __name__ == "__main__":
    main()
