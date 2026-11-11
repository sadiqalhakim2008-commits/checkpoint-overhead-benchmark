"""
Checkpoint Overhead Benchmark - Step 2 (repeated runs)
======================================================
Measures how much periodic checkpointing slows down GPU training.

Question: when we save the model every N steps (so training can resume after
a failure), how much training time do we lose? And does saving in the
background (async) reduce that cost compared to normal (sync) saving?

Run on Google Colab:  Runtime > Change runtime type > T4 GPU
    !python checkpoint_benchmark.py

Each configuration is repeated several times (default 3) so we can report
the mean and the spread (standard deviation) instead of a single noisy number.

Outputs:
    results_raw.csv      - every single run
    results_summary.csv  - mean and std per configuration
    overhead_plot.png    - mean overhead (%) with error bars
"""

import argparse
import csv
import os
import shutil
import threading
import time

import torch
import torch.nn as nn
import torchvision.models as models


# ---------------------------------------------------------------- helpers ---
def get_model(name):
    if name == "resnet18":
        return models.resnet18(num_classes=10)
    if name == "resnet50":
        return models.resnet50(num_classes=10)
        if name == "gpt2":
        from transformers import GPT2Config, GPT2LMHeadModel
        return GPT2LMHeadModel(GPT2Config())
    raise ValueError(f"Unknown model: {name}")


def sync_device(device):
    """Wait for all GPU work to finish so timings are accurate."""
    if device == "cuda":
        torch.cuda.synchronize()


def to_cpu(obj):
    """Deep-copy every tensor in a (nested) state dict to CPU memory."""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj


# -------------------------------------------------------------- benchmark ---
def run(model_name, interval, mode, steps, batch, ckpt_dir, device):
    """
    Train for `steps` steps, saving a checkpoint every `interval` steps.
    interval = 0 means no checkpointing (baseline).
    mode = "sync"  -> training waits while the file is written
    mode = "async" -> copy to CPU, then write the file in a background thread
    """
    torch.manual_seed(0)
    model = get_model(model_name).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    # Synthetic data: isolates checkpoint cost from data-loading cost.
        if model_name == "gpt2":
        x = torch.randint(0, 50257, (batch, 128), device=device)

        def compute_loss():
            return model(x, labels=x).loss
    else:
        x = torch.randn(batch, 3, 224, 224, device=device)
        y = torch.randint(0, 10, (batch,), device=device)

        def compute_loss():
            return loss_fn(model(x), y)

    def train_step():
        opt.zero_grad(set_to_none=True)
        loss = compute_loss()
        loss.backward()
        opt.step()

    for _ in range(5):  # warm-up (first steps are always slower)
        train_step()
    sync_device(device)

    os.makedirs(ckpt_dir, exist_ok=True)
    stall_times, threads, last_path = [], [], None

    t0 = time.perf_counter()
    for i in range(1, steps + 1):
        train_step()

        if interval and i % interval == 0:
            sync_device(device)
            s = time.perf_counter()
            last_path = os.path.join(ckpt_dir, f"{model_name}_{mode}_{i}.pt")
            state = {"step": i, "model": model.state_dict(),
                     "optim": opt.state_dict()}
            if mode == "sync":
                torch.save(state, last_path)
            else:
                snapshot = to_cpu(state)  # training must wait for this copy
                th = threading.Thread(target=torch.save,
                                      args=(snapshot, last_path))
                th.start()
                threads.append(th)
            stall_times.append(time.perf_counter() - s)

    sync_device(device)
    train_time = time.perf_counter() - t0
    for th in threads:  # let background writes finish
        th.join()
    total_with_flush = time.perf_counter() - t0

    size_mb = os.path.getsize(last_path) / 1e6 if last_path else 0.0
    shutil.rmtree(ckpt_dir, ignore_errors=True)  # free disk space

    return {
        "model": model_name,
        "mode": mode if interval else "none",
        "interval": interval,
        "steps": steps,
        "batch": batch,
        "num_checkpoints": len(stall_times),
        "ckpt_size_mb": round(size_mb, 1),
        "avg_stall_s": round(sum(stall_times) / len(stall_times), 4)
        if stall_times else 0.0,
        "train_time_s": round(train_time, 3),
        "total_with_flush_s": round(total_with_flush, 3),
    }


# ------------------------------------------------------------------- main ---
def mean_std(values):
    n = len(values)
    m = sum(values) / n
    sd = (sum((v - m) ** 2 for v in values) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return round(m, 3), round(sd, 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["resnet18", "resnet50"])
    p.add_argument("--intervals", nargs="+", type=int, default=[50, 20, 10])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--ckpt-dir", default="/content/ckpt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else
             "  WARNING: no GPU found - results will not be meaningful"))

    configs = [("none", 0)] + [(mode, itv) for itv in args.intervals
                               for mode in ("sync", "async")]
    raw = []
    for m in args.models:
        for rep in range(1, args.repeats + 1):
            print(f"\n[{m}] repeat {rep}/{args.repeats}")
            base = None
            for mode, itv in configs:
                r = run(m, itv, "sync" if mode == "none" else mode,
                        args.steps, args.batch, args.ckpt_dir, device)
                if mode == "none":
                    base = r["train_time_s"]
                # overhead is measured against the baseline of the SAME repeat
                r["overhead_pct"] = round(
                    100 * (r["train_time_s"] - base) / base, 2)
                r["repeat"] = rep
                raw.append(r)
                print(f"  {mode:5} every {itv:>3} steps | "
                      f"time {r['train_time_s']:7.2f} s | "
                      f"overhead {r['overhead_pct']:6.2f}%")

    with open("results_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(raw[0].keys()))
        w.writeheader()
        w.writerows(raw)

    summary = []
    for m in args.models:
        for mode, itv in configs:
            rows = [r for r in raw if r["model"] == m
                    and r["mode"] == mode and r["interval"] == itv]
            ov_m, ov_sd = mean_std([r["overhead_pct"] for r in rows])
            st_m, st_sd = mean_std([r["avg_stall_s"] for r in rows])
            summary.append({
                "model": m, "mode": mode, "interval": itv,
                "repeats": len(rows),
                "overhead_mean_pct": ov_m, "overhead_std_pct": ov_sd,
                "stall_mean_s": st_m, "stall_std_s": st_sd,
                "ckpt_size_mb": rows[0]["ckpt_size_mb"],
            })

    with open("results_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print("\n=== SUMMARY (mean +- std over", args.repeats, "repeats) ===")
    for s_ in summary:
        if s_["mode"] == "none":
            continue
        print(f"{s_['model']:8} | {s_['mode']:5} | every {s_['interval']:>3} | "
              f"overhead {s_['overhead_mean_pct']:6.2f}% +- "
              f"{s_['overhead_std_pct']:.2f}")
    print("\nSaved results_raw.csv and results_summary.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for m in args.models:
            for mode, style in (("sync", "-o"), ("async", "--s")):
                pts = sorted((s_["interval"], s_["overhead_mean_pct"],
                              s_["overhead_std_pct"]) for s_ in summary
                             if s_["model"] == m and s_["mode"] == mode)
                if pts:
                    xs, ys, es = zip(*pts)
                    ax.errorbar(xs, ys, yerr=es, fmt=style, capsize=4,
                                label=f"{m} ({mode})")
        ax.set_xlabel("Checkpoint interval (steps)")
        ax.set_ylabel("Training time overhead (%)")
        ax.set_title(f"Cost of checkpointing during GPU training "
                     f"(mean +- std, n={args.repeats})")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig("overhead_plot.png", dpi=150)
        print("Saved overhead_plot.png")
    except ImportError:
        print("matplotlib not installed - skipped plot")


if __name__ == "__main__":
    main()
