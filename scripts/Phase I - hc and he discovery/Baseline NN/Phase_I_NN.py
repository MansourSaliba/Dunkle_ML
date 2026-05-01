"""
Phase I - Feed-Forward NN for joint prediction of he and hc

- Inputs: Tw, Tgi
- Outputs: he, hc (single neural network)
- Loss: supervised data MSE only (no physics residual term)

- Data split: sequential 70% train, 20% validation, 10% test
- Test metrics are computed on the held-out 10% split.
- Also runs a full time-ordered forward pass over dunkle_clean and
    overlays predicted he/hc against ground truth.
"""

from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise ImportError(
        "PyTorch is not installed. Install it with: pip install torch"
    ) from exc


TRAIN_FRAC = 0.70
VAL_FRAC = 0.20
EPS = 1e-12

# Training hyperparameters
EPOCHS = 3000
LR = 1e-3
WEIGHT_DECAY = 1e-6
PATIENCE = 300
HIDDEN = 64
LAYERS = 3
SEED = 42
HC_NEG_PENALTY = 0


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(base_dir: Path) -> pd.DataFrame:
    path = base_dir / "dunkle_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.name}")
    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame) -> None:
    required = {"Tw", "Tgi", "he", "hc"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def split_indices(n: int) -> tuple[slice, slice, slice]:
    n_train = int(TRAIN_FRAC * n)
    n_val = int(VAL_FRAC * n)
    n_test = n - n_train - n_val
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n_train + n_val + n_test)


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


class JointNN(nn.Module):
    def __init__(self, in_dim: int = 2, hidden: int = HIDDEN, layers: int = LAYERS, out_dim: int = 2):
        super().__init__()
        modules: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            modules.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        modules.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(mse(a, b)))


def r2(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.sum((a - np.mean(a)) ** 2))
    if denom <= 0.0:
        return float("nan")
    return float(1.0 - np.sum((a - b) ** 2) / denom)


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = Path(__file__).resolve().parent
    processed_dir = Path(__file__).resolve().parents[3] / "data" / "processed"
    df = load_data(processed_dir)
    validate_columns(df)

    # Keep the full sequence order for sequential split.
    x = df[["Tw", "Tgi"]].to_numpy(dtype=np.float64)
    y = df[["he", "hc"]].to_numpy(dtype=np.float64)

    train_slice, val_slice, test_slice = split_indices(len(df))
    x_train, x_val, x_test = x[train_slice], x[val_slice], x[test_slice]
    y_train, y_val, y_test = y[train_slice], y[val_slice], y[test_slice]

    # Normalize using train statistics only.
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std = np.where(x_std < EPS, 1.0, x_std)

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_std = np.where(y_std < EPS, 1.0, y_std)

    x_train_n = (x_train - x_mean) / x_std
    x_val_n = (x_val - x_mean) / x_std
    x_test_n = (x_test - x_mean) / x_std

    y_train_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std

    # Tensors
    X_train = to_tensor(x_train_n, device)
    X_val = to_tensor(x_val_n, device)
    X_test = to_tensor(x_test_n, device)

    Y_train = to_tensor(y_train_n, device)
    Y_val = to_tensor(y_val_n, device)

    model = JointNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = math.inf
    best_state = None
    wait = 0

    history: list[dict[str, float]] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        pred_train_n = model(X_train)
        data_loss_train = ((pred_train_n - Y_train) ** 2).mean()

        pred_train = pred_train_n * to_tensor(y_std, device) + to_tensor(y_mean, device)
        hc_train = pred_train[:, 1]
        neg_penalty = torch.clamp(-hc_train, min=0.0).pow(2).mean()

        total_train = data_loss_train + HC_NEG_PENALTY * neg_penalty
        total_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val_n = model(X_val)
            data_loss_val = ((pred_val_n - Y_val) ** 2).mean()

            total_val = data_loss_val

        val_total = float(total_val.item())
        history.append(
            {
                "epoch": epoch,
                "train_data_loss": float(data_loss_train.item()),
                "train_phys_loss": 0.0,
                "train_total_loss": float(total_train.item()),
                "val_data_loss": float(data_loss_val.item()),
                "val_phys_loss": 0.0,
                "val_total_loss": val_total,
            }
        )

        if val_total < best_val:
            best_val = val_total
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if wait >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Train/validation/test metrics in physical units.
    model.eval()
    with torch.no_grad():
        pred_train_n = model(X_train).cpu().numpy()
        pred_val_n = model(X_val).cpu().numpy()
        pred_test_n = model(X_test).cpu().numpy()

    pred_train = pred_train_n * y_std + y_mean
    pred_val = pred_val_n * y_std + y_mean
    pred_test = pred_test_n * y_std + y_mean

    # Full-sequence forward pass in time order.
    x_full_n = (x - x_mean) / x_std
    X_full = to_tensor(x_full_n, device)
    with torch.no_grad():
        pred_full_n = model(X_full).cpu().numpy()
    pred_full = pred_full_n * y_std + y_mean

    he_train_true, hc_train_true = y_train[:, 0], y_train[:, 1]
    he_val_true, hc_val_true = y_val[:, 0], y_val[:, 1]
    he_test_true, hc_test_true = y_test[:, 0], y_test[:, 1]
    he_full_true, hc_full_true = y[:, 0], y[:, 1]
    he_train_pred, hc_train_pred = pred_train[:, 0], pred_train[:, 1]
    he_val_pred, hc_val_pred = pred_val[:, 0], pred_val[:, 1]
    he_test_pred, hc_test_pred = pred_test[:, 0], pred_test[:, 1]
    he_full_pred, hc_full_pred = pred_full[:, 0], pred_full[:, 1]

    metrics = {
        "he_train_mse": mse(he_train_true, he_train_pred),
        "he_val_mse": mse(he_val_true, he_val_pred),
        "he_test_mse": mse(he_test_true, he_test_pred),
        "he_train_rmse": rmse(he_train_true, he_train_pred),
        "he_val_rmse": rmse(he_val_true, he_val_pred),
        "he_test_rmse": rmse(he_test_true, he_test_pred),
        "he_train_r2": r2(he_train_true, he_train_pred),
        "he_val_r2": r2(he_val_true, he_val_pred),
        "he_test_r2": r2(he_test_true, he_test_pred),
        "he_full_mse": mse(he_full_true, he_full_pred),
        "he_full_rmse": rmse(he_full_true, he_full_pred),
        "he_full_r2": r2(he_full_true, he_full_pred),
        "hc_train_mse": mse(hc_train_true, hc_train_pred),
        "hc_val_mse": mse(hc_val_true, hc_val_pred),
        "hc_test_mse": mse(hc_test_true, hc_test_pred),
        "hc_train_rmse": rmse(hc_train_true, hc_train_pred),
        "hc_val_rmse": rmse(hc_val_true, hc_val_pred),
        "hc_test_rmse": rmse(hc_test_true, hc_test_pred),
        "hc_train_r2": r2(hc_train_true, hc_train_pred),
        "hc_val_r2": r2(hc_val_true, hc_val_pred),
        "hc_test_r2": r2(hc_test_true, hc_test_pred),
        "hc_full_mse": mse(hc_full_true, hc_full_pred),
        "hc_full_rmse": rmse(hc_full_true, hc_full_pred),
        "hc_full_r2": r2(hc_full_true, hc_full_pred),
        "train_points": int(len(y_train)),
        "val_points": int(len(y_val)),
        "test_points": int(len(y_test)),
        "best_val_loss": float(best_val),
        "epochs_ran": int(len(history)),
    }

    # Save artifacts.
    hist_df = pd.DataFrame(history)
    hist_path = base_dir / "phase1_nn_history.csv"
    hist_df.to_csv(hist_path, index=False)

    metrics_df = pd.DataFrame([metrics])
    metrics_path = base_dir / "phase1_nn_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Save full-sequence forward predictions.
    if "time_s" in df.columns:
        time_axis = df["time_s"].to_numpy(dtype=float)
        time_col = "time_s"
    else:
        time_axis = np.arange(len(df), dtype=float)
        time_col = "index"

    pred_df = pd.DataFrame(
        {
            time_col: time_axis,
            "he_true": he_full_true,
            "he_pred": he_full_pred,
            "hc_true": hc_full_true,
            "hc_pred": hc_full_pred,
        }
    )
    pred_path = base_dir / "phase1_nn_forward_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    model_path = base_dir / "phase1_nn_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "config": {
                "epochs": EPOCHS,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "patience": PATIENCE,
                "hidden": HIDDEN,
                "layers": LAYERS,
                "seed": SEED,
            },
        },
        model_path,
    )

    # Loss plot.
    plt.figure(figsize=(10, 6))
    plt.plot(hist_df["epoch"], hist_df["train_total_loss"], label="train_total_loss", lw=1.8)
    plt.plot(hist_df["epoch"], hist_df["val_total_loss"], label="val_total_loss", lw=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("NN Training Progress")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    loss_plot_path = base_dir / "phase1_nn_losses.png"
    plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Overlay plots: predicted vs ground truth over time.
    he_plot_path = base_dir / "phase1_nn_forward_overlay_he.png"
    plt.figure(figsize=(10, 5.5))
    plt.plot(time_axis, he_full_true, color="tab:blue", lw=2.0, label="he ground")
    plt.plot(time_axis, he_full_pred, color="tab:red", lw=1.8, label="he predicted")
    plt.xlabel(time_col)
    plt.ylabel("he")
    plt.title("NN Forward Comparison - he")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(he_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    hc_plot_path = base_dir / "phase1_nn_forward_overlay_hc.png"
    plt.figure(figsize=(10, 5.5))
    plt.plot(time_axis, hc_full_true, color="tab:blue", lw=2.0, label="hc ground")
    plt.plot(time_axis, hc_full_pred, color="tab:red", lw=1.8, label="hc predicted")
    plt.xlabel(time_col)
    plt.ylabel("hc")
    plt.title("NN Forward Comparison - hc")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(hc_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("=" * 72)
    print("Phase I NN complete")
    print(f"Rows total: {len(df)}")
    print(f"Split -> train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")
    print(f"History saved: {hist_path.name}")
    print(f"Metrics saved: {metrics_path.name}")
    print(f"Model saved: {model_path.name}")
    print(f"Loss plot saved: {loss_plot_path.name}")
    print(f"Forward predictions saved: {pred_path.name}")
    print(f"Forward he overlay saved: {he_plot_path.name}")
    print(f"Forward hc overlay saved: {hc_plot_path.name}")
    print(f"he test RMSE: {metrics['he_test_rmse']:.6e}")
    print(f"hc test RMSE: {metrics['hc_test_rmse']:.6e}")
    print(f"he full RMSE: {metrics['he_full_rmse']:.6e}")
    print(f"hc full RMSE: {metrics['hc_full_rmse']:.6e}")
    print("=" * 72)


if __name__ == "__main__":
    main()
