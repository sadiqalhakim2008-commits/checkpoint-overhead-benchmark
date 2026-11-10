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
- Models: ResNet-18 and ResNet-50
- Data: random tensors (224x224, batch size 32), so data loading does not affect the measurement
- 200 training steps per run
- Checkpoint every 50, 20 and 10 steps
- Checkpoint size: ~90 MB (ResNet-18), ~189 MB (ResNet-50), including SGD momentum

Overhead = extra training time compared to the same run without checkpointing.

## Results (first run)

| Model | Interval | Sync overhead | Async overhead |
|---|---|---|---|
| ResNet-18 | 50 | 5.78% | 7.62% |
| ResNet-18 | 20 | 18.72% | 19.48% |
| ResNet-18 | 10 | 29.56% | 14.17% |
| ResNet-50 | 50 | 1.53% | 0.08% |
| ResNet-50 | 20 | 3.70% | 0.87% |
| ResNet-50 | 10 | 11.99% | 1.47% |

These numbers are from a single run. I am now repeating each configuration 3 times to check how stable they are.

## Observations

- Overhead grows fast when checkpoints are frequent. With sync saving every 10 steps, ResNet-18 training became about 30% slower.
- For ResNet-50, async saving almost removed the cost: at a 10-step interval the overhead dropped from about 12% to about 1.5%.
- For ResNet-18, async did not help at 50 and 20 steps and was even slightly worse than sync, even though the time training was blocked per checkpoint was shorter (about 0.05 s vs 0.13 s).

My current explanation: ResNet-18 steps are short (about 88 ms per step), so the background write overlaps with training and they compete for CPU, memory bandwidth and the Python GIL. ResNet-50 steps are about 4 times longer (about 343 ms), so the write can finish with less interference. So async saving seems to be worth it when each training step takes much longer than writing the checkpoint. I still need more experiments to confirm this.

## Limitations

- One GPU only. Real HPC systems save checkpoints from many nodes to parallel file systems.
- Synthetic data only.
- Results table above is from one run.

## Next steps

- Repeat each configuration and report mean and standard deviation.
- Find out which part of the async path causes the slowdown (CPU copy, disk write, or GIL).
- Try larger models and different storage (local disk vs. network storage).

## Running it

On Google Colab with a T4 GPU:

```bash
!python checkpoint_benchmark.py              # 3 repeats, about 35 min
!python checkpoint_benchmark.py --repeats 1  # quick run
```

Output files: `results_raw.csv`, `results_summary.csv`, `overhead_plot.png`.

Code written with help from AI coding tools.

## Author

Sadek Al-Hakeem, Automation and AI Engineering, Al-Nahrain University, Baghdad
