""" 

This code was taken from the Github Repo: https://github.com/hoangthangta/All-KAN/blob/main/run_ff.py

and edited for new W-CSRBF Models and regularizers

"""
import argparse, os, time, types
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
from models import (
    EfficientKAN, FastKAN, BSRBF_KAN, FasterKAN, MLP, FC_KAN, WCSRBFKAN, WCSRBFKANSolo
)
from file_io import *

def exp_sin(x1, x2):
    # e^{sin(x1^2 + x2^2)}
    return torch.exp(torch.sin(x1**2 + x2**2))

def exp_jo(x1, x2):
    # e^{J0(20 x1) + x2^2}
    return torch.exp(torch.special.bessel_j0(20.0 * x1) + x2**2)

def div(x1, x2, eps=1e-3):
    # x1 / x2   (avoid div-by-zero)
    return x1 / (x2 + eps)

def sum_prod(x1, x2):
    # (x1 + x2) + x1*x2
    return x1 + x2 + x1 * x2

def franke2d(x1, x2):
    """
    Franke's function on x in [0,1]:
      0.75*exp(-((9x-2)^2)/4 - ((9y-2)^2)/4)
    + 0.75*exp(-((9x+1)^2)/49 - (9y+1)/10)
    + 0.5 *exp(-((9x-7)^2)/4 - ((9y-3)^2)/4)
    - 0.2 *exp(-(9x-4)^2 - (9y-7)^2)
    """
    x, y = x1, x2
    term1 = 0.75 * torch.exp(-((9*x - 2)**2)/4.0 - ((9*y - 2)**2)/4.0)
    term2 = 0.75 * torch.exp(-((9*x + 1)**2)/49.0 - ((9*y + 1)/10.0))
    term3 = 0.5  * torch.exp(-((9*x - 7)**2)/4.0 - ((9*y - 3)**2)/4.0)
    term4 = -0.2 * torch.exp(-((9*x - 4)**2)      - ((9*y - 7)**2))
    return term1 + term2 + term3 + term4


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def pick_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_model(name, layers):
    if name == 'fc_kan':
        return FC_KAN(layers, func_list=["dog","bs"], layernorm=False)
    elif name == 'efficient_kan':
        return EfficientKAN(layers)
    elif name == 'bsrbf_kan':
        return BSRBF_KAN(layers, layernorm=False)
    elif name == 'fast_kan':
        return FastKAN(layers, layernorm=False)
    elif name == 'faster_kan':
        return FasterKAN(layers, layernorm=False)
    elif name == 'mlp':
        return MLP(layers, layernorm=False)
    elif(name == 'wcsrbf_kan_un'):
        return WCSRBFKANSolo(layers, enable_layer_norm=False, trainable_centers=False, trainable_sigma=False)
    elif(name == 'wcsrbf_kan_tc_ts_un'):
        return WCSRBFKANSolo(layers, enable_layer_norm=False, trainable_centers=True, trainable_sigma=True)
    elif(name == 'wcsrbf_kan_ts_un'):
        return WCSRBFKANSolo(layers, enable_layer_norm=False, trainable_centers=False, trainable_sigma=True)
    elif(name == 'wcsrbf_kan_tc_un'):
        return WCSRBFKANSolo(layers, enable_layer_norm=False, trainable_centers=True, trainable_sigma=False)
    elif(name == 'wcsrbf_kan'):
        return WCSRBFKANSolo(layers, enable_layer_norm=True, trainable_centers=False, trainable_sigma=False)
    elif(name == 'wcsrbf_kan_tc_ts'):
        return WCSRBFKANSolo(layers, enable_layer_norm=True, trainable_centers=True, trainable_sigma=True)
    elif(name == 'wcsrbf_kan_ts'):
        return WCSRBFKANSolo(layers, enable_layer_norm=True, trainable_centers=False, trainable_sigma=True)
    elif(name == 'wcsrbf_kan_tc'):
        return WCSRBFKANSolo(layers, enable_layer_norm=True, trainable_centers=True, trainable_sigma=False)
    elif(name == 'wcsrbf_kan_solo'):
        return WCSRBFKANSolo(layers, enable_layer_norm=False, trainable_centers=False, trainable_sigma=False, use_base=False)
    elif(name == 'wcsrbf_kan_solo_tc'):
        return WCSRBFKANSolo(layers, enable_layer_norm=False, trainable_centers=True, trainable_sigma=False, use_base=False)
    elif(name == 'wcsrbf_kan_solo_ts'):
        return WCSRBFKANSolo(layers, enable_layer_norm=False, trainable_centers=False, trainable_sigma=True, use_base=False)
    elif(name == 'wcsrbf_kan_solo_tc_ts'):
        return WCSRBFKANSolo(layers, enable_layer_norm=False, trainable_centers=True, trainable_sigma=True, use_base=False)
    else:
        raise ValueError(f"Unsupported model '{name}'")

# meshgrid helpers for 2D
def make_grid_from_data(x_np, delta):
    x1_min, x1_max = np.round(x_np[:,0].min(), 3), np.round(x_np[:,0].max(), 3)
    x2_min, x2_max = np.round(x_np[:,1].min(), 3), np.round(x_np[:,1].max(), 3)
    x1 = np.arange(x1_min, x1_max + delta, delta)
    x2 = np.arange(x2_min, x2_max + delta, delta)
    X1, X2 = np.meshgrid(x1, x2)
    return X1, X2, (x1_min, x1_max, x2_min, x2_max)

@torch.no_grad()
def eval_model_on_grid(model, X1, X2, device):
    # [H,W] -> flatten -> model -> reshape back
    X = np.column_stack((X1.ravel(), X2.ravel()))
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    Y = model(Xt)
    # be robust to model output shapes
    if Y.ndim == 3:  # (N, out, ?)
        Y = Y[:,:,0]
    if Y.ndim == 2 and Y.shape[1] == 1:
        Y = Y[:,0]
    Y = Y.detach().cpu().numpy().reshape(X1.shape)
    return Y

@torch.no_grad()
def eval_func_on_grid(func_torch, X1, X2, device):
    X1t = torch.tensor(X1, dtype=torch.float32, device=device)
    X2t = torch.tensor(X2, dtype=torch.float32, device=device)
    Yt = func_torch(X1t, X2t)
    if Yt.ndim == 2 and Yt.shape[1] == 1:
        Yt = Yt[:,0]
    Y = Yt.detach().cpu().numpy()
    return Y

def save_imshow(Y, ex, title, path):
    plt.figure()
    im = plt.imshow(
        Y,
        interpolation='bilinear',
        cmap=cm.RdYlGn,
        origin='lower',
        extent=[ex[0], ex[1], ex[2], ex[3]],
        vmax=np.abs(Y).max(),
        vmin=-np.abs(Y).max()
    )
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

def plot_and_save_meshgrids(x_train_np, gt_func, model, delta, out_dir, device, tag):
    X1, X2, ex = make_grid_from_data(x_train_np, delta)
    Y_gt = eval_func_on_grid(gt_func, X1, X2, device)
    Y_pred = eval_model_on_grid(model, X1, X2, device)
    err = np.abs(Y_pred - Y_gt)

    save_imshow(Y_gt, ex, f"Ground Truth ({tag})", os.path.join(out_dir, f"{tag}_gt.png"))
    save_imshow(Y_pred, ex, f"Prediction ({tag})", os.path.join(out_dir, f"{tag}_pred.png"))
    plt.figure()
    plt.imshow(err, origin='lower', extent=[ex[0], ex[1], ex[2], ex[3]])
    plt.title(f"Abs Error ({tag})")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_err.png"), dpi=180)
    plt.close()

def make_data_for_func(args, device):
    """
    Returns:
      x : torch.FloatTensor [N,D]
      y : torch.FloatTensor [N,1]
      gt_func_for_grid : callable(x1, x2) -> torch tensor (only for 2D funcs)
      tag : short name for saving
    """
    # unified registry
    registry = {
        "exp_sin": {
            "dims":2,
            "call":lambda x: exp_sin(x[:,[0]], x[:,[1]]),
            "grid":lambda X1,X2: exp_sin(X1, X2)
        },
        "exp_jo": {
            "dims":2,
            "call":lambda x: exp_jo(x[:,[0]], x[:,[1]]),
            "grid":lambda X1,X2: exp_jo(X1, X2)
        },
        "div": {
            "dims":2,
            "call":lambda x: div(x[:,[0]], x[:,[1]]),
            "grid":lambda X1,X2: div(X1, X2)
        },
        "sum_prod": {
            "dims":2,
            "call":lambda x: sum_prod(x[:,[0]], x[:,[1]]),
            "grid":lambda X1,X2: sum_prod(X1, X2)
        },
        "franke2d":  {
            "dims":2, 
            "domain": (0.0, 1.0),
            "call":lambda x: franke2d(x[:,[0]], x[:,[1]]),
            "grid":lambda X1,X2: franke2d(X1, X2)
        },
    }

    if args.func not in registry:
        raise ValueError(f"Unknown func '{args.func}'. Choices: {list(registry.keys())}")

    spec = registry[args.func]
    D = spec["dims"]

    if D == 2:
        a1, b1 = spec.get("domain", (args.range_min, args.range_max))
        a2, b2 = spec.get("domain", (args.range_min, args.range_max))
        x = torch.empty((args.n_samples, 2), device=device)
        x[:,0] = (b1 - a1) * torch.rand(args.n_samples, device=device) + a1
        x[:,1] = (b2 - a2) * torch.rand(args.n_samples, device=device) + a2
        y = spec["call"](x)
        gt = spec["grid"]
        return x, y, gt, args.func
    
def weights_l2(model_name, model, lambda_w=1e-4):
    reg = 0.0
    if model_name == "mlp":
        for l in model.layers:
            reg += l.base_weight.pow(2).sum()
        return lambda_w * reg
    if model_name == "efficient_kan":
        for l in model.layers:
            reg += l.base_weight.pow(2).sum() + l.spline_weight.pow(2).sum() + l.spline_scaler.pow(2).sum()
        return lambda_w * reg
    if model_name == "fast_kan":
        for l in model.layers:
            reg += l.base_linear.weight.pow(2).sum() + l.base_linear.bias.pow(2).sum() + l.spline_linear.weight.pow(2).sum()
        return lambda_w * reg
    if model_name == "faster_kan":
        for l in model.layers:
            reg += l.spline_linear.weight.pow(2).sum()
        return lambda_w * reg
    if model_name == "bsrbf_kan":
        for l in model.layers:
            reg += l.base_weight.pow(2).sum() + l.spline_weight.pow(2).sum()
        return lambda_w * reg
    if model_name == "fc_kan":
        for l in model.layers:
            reg += l.base_weight.pow(2).sum() + l.spline_weight.pow(2).sum()
        return lambda_w * reg
    else: 
        return reg

def train(args):
    device = pick_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Parse layers
    layers = [int(x) for x in args.layers.split(',')]
    model = build_model(args.model_name, layers).to(device)

    # Data
    x, y, gt_grid_func, tag = make_data_for_func(args, device)

    if x.shape[1] != layers[0]:
        raise ValueError(f"Expected input dim {layers[0]}, got {x.shape[1]}")

    # Optim & loss
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = torch.nn.MSELoss()

    run_dir = ensure_dir(os.path.join(args.save_dir, tag, args.model_name))
    json_file = os.path.join(run_dir, "training.json")

    # Train loop
    t0 = time.time()
    losses = []
    for e in range(args.epochs):
        model.train()
        opt.zero_grad()
        pred = model(x)
        # robust to different outputs
        if pred.ndim == 3:
            pred = pred[:,:,0]
        if pred.ndim == 1:
            pred = pred.unsqueeze(1)
        loss = mse(pred, y)
        if "wcsrbf" in args.model_name:
            if model.trainable_sigma_bool:
                loss += model.sigma_inverse_l2(1e-3)
            loss += model.weights_l2(1e-3)
        else:
            loss += weights_l2(args.model_name, model, 1e-3)
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if (e+1) % max(1, args.epochs//10) == 0:
            print(f"[{e+1}/{args.epochs}] loss={loss.item():.6f}")
            write_single_dict_to_jsonl(json_file, { "epoch": e+1, "loss": loss.item()}, file_access = 'a')

    t1 = time.time()
    print(f"Training time: {t1 - t0:.2f}s | Final loss: {losses[-1]:.6e}")
    write_single_dict_to_jsonl(json_file, { "training_time:": f"{t1 - t0:.4f}s", "final_loss": losses[-1]}, file_access = 'a')

    # Save loss curve
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curve.png"), dpi=180)
    plt.close()

    # Save model
    torch.save({"model_state": model.state_dict(), "layers": layers}, os.path.join(run_dir, "model.pt"))

    # Meshgrid plots for 2D functions
    if x.shape[1] == 2 and gt_grid_func is not None:
        model.eval()
        x_np = x.detach().cpu().numpy()
        plot_and_save_meshgrids(
            x_train_np=x_np,
            gt_func=gt_grid_func,
            model=model,
            delta=args.delta,
            out_dir=run_dir,
            device=device,
            tag=tag
        )
        print(f"Saved meshgrids to: {run_dir}")

def main():
    p = argparse.ArgumentParser(description="Train and plot meshgrid for 2D functions")
    p.add_argument('--mode', type=str, default='train')
    p.add_argument('--model_name', type=str, default='efficient_kan')
    p.add_argument('--layers', type=str, default='2,1')
    p.add_argument('--func', type=str, default='exp_sin')
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--n_samples', type=int, default=8000)
    p.add_argument('--range_min', type=float, default=-3.0)
    p.add_argument('--range_max', type=float, default=3.0)
    p.add_argument('--delta', type=float, default=0.02)
    p.add_argument('--save_dir', type=str, default='outputs_ff')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    if args.mode == 'train':
        train(args)
    else:
        raise ValueError("Only --mode train is supported in this script.")

if __name__ == '__main__':
    main()

#python function_fitting.py --mode "train" --model_name "wcsrbf_kan" --layers "2,5,1" --func "exp_sin" --epochs 1000 --save_dir "output_ff/";