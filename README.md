# Checkpoint Overhead Benchmark

Measuring how much periodic checkpointing slows down GPU training, and comparing synchronous vs. asynchronous checkpoint saving.

## Motivation

Long training jobs on large clusters can crash because of hardware failures, so they save checkpoints (model + optimizer state) regularly to be able to resume. Saving takes time though. I wanted to see how big this cost is in practice and whether saving in a background thread reduces it.

Two strategies are compared:

- **sync**: training stops until the checkpoint is written to disk.
- **async**: the state is copied to CPU memory first, then written to disk by a background thread while training continues.

## Setup

- GPU: NVIDIA Tesla T4 (Google Colab)
- Framework: PyTorch
- Models: ResNet-18 (~97 ms per step) and ResNet-50 (~322 ms per step)
- Data: random tensors (224x224, batch size 32), so data loading does not affect the measurement
- 200 training steps per run, each configuration repeated 3 times
- Checkpoint every 50, 20 and 10 steps
- Checkpoint size: ~90 MB (ResNet-18), ~189 MB (ResNet-50), including SGD momentum

Overhead = extra training time compared to a run without checkpointing (baseline measured in the same repeat).

## Results (mean ± std, 3 repeats)

Training time overhead:

| Model | Interval (steps) | Sync | Async |
|---|---|---|---|
| ResNet-18 | 50 | 4.18% ± 1.62 | 0.94% ± 0.42 |
| ResNet-18 | 20 | 7.01% ± 0.29 | 1.98% ± 0.43 |
| ResNet-18 | 10 | 26.25% ± 9.21 | 4.53% ± 1.11 |
| ResNet-50 | 50 | 1.89% ± 0.61 | 0.08% ± 0.56 |
| ResNet-50 | 20 | 5.21% ± 0.49 | 0.95% ± 0.66 |
| ResNet-50 | 10 | 25.67% ± 15.81 | 1.97% ± 0.28 |

Time training is blocked per checkpoint: about 0.15 s (sync) vs 0.04 s (async) for ResNet-18, and about 0.35 s (sync) vs 0.07 s (async) for ResNet-50.

![Overhead plot](overhead_plot.png)

Raw data: [`results/results_raw.csv`](results/results_raw.csv), summary: [`results/results_summary.csv`](results/results_summary.csv).

## Observations

1. **Async saving reduced overhead for both models at every interval.** At a 10-step interval, overhead dropped from ~26% to ~4.5% (ResNet-18) and from ~26% to ~2% (ResNet-50).

2. **Sync saving becomes unstable when checkpoints are frequent.** At a 10-step interval the standard deviation is very large (± 9 and ± 16 points). In one ResNet-50 run the overhead reached 43%, and the blocked time per checkpoint grew from ~0.35 s to ~1.4 s. My guess is that the storage cannot absorb writes this fast, so later writes have to wait for earlier ones to be flushed.

3. **Async has a hidden cost: write backlog.** For ResNet-18 at a 10-step interval, training finished on time, but in one run the background writes needed **27 extra seconds** to finish after training ended (2.9 s in another run, under 0.2 s in the rest). That interval produces roughly 90 MB of checkpoint data per second, which seems to be more than the disk could sustain. ResNet-50 at the same interval produces about 57 MB/s and never built a backlog (≤ 0.4 s). While a backlog exists, the newest checkpoints are not yet on disk, so a crash at that moment would lose more work than expected, and the queued snapshots also use CPU memory.

4. **A single run was misleading.** My first single-run test suggested async was slightly worse than sync for ResNet-18 at long intervals. With 3 repeats this did not hold: async was better in every configuration. The earlier result was within run-to-run noise.

**Takeaway:** async checkpointing hides most of the cost, but only as long as the checkpoint data rate stays below what the storage can write. Above that, the cost moves from "training is slower" to "checkpoints are late".

## GPT-2 small (large checkpoint)

To see what happens with a much bigger checkpoint, I repeated the test with GPT-2 small (124M parameters, random weights, batch size 8, sequence length 128, 200 steps, about 0.28 s per step), checkpointing every 50 steps. The checkpoint is about 1 GB (model + SGD momentum). I ran the whole test twice, 3 repeats each, in two different Colab sessions: first with my original async copy, then after fixing a bug in it (point 2 below).

After the fix:

| Mode | Overhead during training | Blocked time per checkpoint | Time until all checkpoints are on disk, vs. no checkpointing |
|---|---|---|---|
| sync | 100.9% ± 54.2 (72%, 67%, 163%) | 13.9 s ± 8.8 | same as training time |
| async | 8.4% ± 8.8 (18.5%, 3.4%, 3.2%) | 0.49 s ± 0.07 | +42%, +22%, +73% |

Before the fix (first run):

| Mode | Overhead during training | Blocked time per checkpoint | Time until all checkpoints are on disk, vs. no checkpointing |
|---|---|---|---|
| sync | 131.5% ± 63.0 (199%, 74%, 122%) | 18.7 s ± 7.8 | same as training time |
| async | 7.1% ± 4.8 (11.5%, 7.9%, 2.0%) | 0.57 s ± 0.27 | +275%, +203%, +24% |

Raw data after the fix: [`results/gpt2_results_fixed_raw.csv`](results/gpt2_results_fixed_raw.csv), [`results/gpt2_results_fixed_summary.csv`](results/gpt2_results_fixed_summary.csv). Before the fix: [`results/gpt2_results_raw.csv`](results/gpt2_results_raw.csv), [`results/gpt2_results_summary.csv`](results/gpt2_results_summary.csv).

What I see:

1. **Async hides almost all of the blocking.** In both sessions, training itself was only 2-19% slower with async, compared with 67-199% for sync. One synchronous save stopped training for 8-27 s (about 30-95 training steps' worth of time), one async save for about 0.5 s.

2. **I found and fixed a bug in my async copy.** In the first run the async checkpoint was 1150 MB while the sync one was 996 MB. The 154 MB difference is exactly GPT-2's embedding matrix (50257 x 768 x 4 bytes). GPT-2 uses the same matrix for the input embedding and the output layer, `torch.save` stores it once, but my CPU copy made two separate copies, so it was written twice. A separate check confirmed this: the original state dict is 497.8 MB, a plain copy is 652.2 MB, and a copy that keeps the sharing is 497.8 MB again. `to_cpu()` now keeps shared tensors shared, and both modes write 995.6 MB.

3. **The write backlog is smaller after the fix, but still there.** The background writer was still working 12, 11 and 41 s after training ended (before the fix: 146, 115 and 13 s). After the fix, async had all checkpoints on disk sooner than sync in all three runs (72 s vs 88 s, 71 s vs 98 s, 101 s vs 154 s); before the fix it was slower in two of three. I cannot say the fix alone caused this: it removed 13% of the data, but the change is larger than that, and sync was also faster in the second session (13.9 s blocked per checkpoint instead of 18.7 s), so the disk was probably faster too. I did not control for that.

4. **Disk speed varies a lot, and so do the results.** Dividing the 996 MB sync checkpoint by its blocked time gives an effective write speed of 129, 100 and 42 MB/s in the second session and 37, 90 and 55 MB/s in the first. The no-checkpoint baseline also varied between 51 and 59 s. With 3 repeats I can say that async blocks training far less, but not give precise percentages for either mode.

## Limitations

- One GPU and one local disk. Real HPC systems save from many nodes to parallel file systems, where bandwidth is shared.
- Synthetic data only.
- Only 3 repeats; the high-variance configurations (10-step ResNet, all GPT-2 sync runs) would need more.
- GPT-2 was only tested at a 50-step interval, in two separate Colab sessions whose disk speed differed.
- The storage explanations above are hypotheses; I did not measure disk throughput directly.

## Next steps

- Measure the disk write bandwidth directly and check whether the backlog starts exactly when the checkpoint data rate exceeds it.
- Limit the async queue (e.g. skip or wait if the previous checkpoint is still being written) and measure the trade-off.
- Try more intervals for GPT-2 and different storage (local disk vs. Google Drive / network storage).
- Save only part of the state (for example weights without optimizer buffers, or only the layers that changed, as in LLMTailor) and check whether smaller checkpoints avoid the backlog.

## Related work

This is a small single-GPU experiment. Two papers study the same problem at real scale:

- Xu et al., "An Efficient Checkpointing System for Large Machine Learning Model Training" (SC-W 2024). Checkpoints are first written to node-local storage and then moved to the shared file system asynchronously. The authors note that insufficient node-local memory can overflow and interrupt training, which is the large-scale version of the backlog I observed here.
- Sun et al., "LLMTailor: A Layer-wise Tailoring Tool for Efficient Checkpointing of Large Language Models" (PDSW 2025). Only the layers that changed significantly are checkpointed, which reduces checkpoint size and time.

## Running it

On Google Colab with a T4 GPU:

```bash
!python checkpoint_benchmark.py              # 3 repeats, about 35 min
!python checkpoint_benchmark.py --repeats 1  # quick run
!python checkpoint_benchmark.py --models gpt2 --intervals 50 --batch 8  # GPT-2, about 10 min
```

Output files: `results_raw.csv`, `results_summary.csv`, `overhead_plot.png`.

Code written with help from AI coding tools.

## Author

Sadeq Al-Hakeem, Automation and AI Engineering, Al-Nahrain University, Baghdad
