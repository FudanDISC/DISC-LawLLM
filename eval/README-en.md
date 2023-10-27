# DISC-Law-Eval Benchmark

- Evaluation datasets: [objective](./datasets/objective/) and [subjective](./datasets/subjective/).

- [Evaluation scripts](./src/): run `src/main.py` for evaluation. Model settings are in `src/models.py` and the paths must be customized. Few-shot examples for objective evaluation can be found [here](./src/few_shot/). When running the evaluation script, model responses will be stored under `responses/` (of the same format as the corresponding evaluation dataset) and evaluation results will be stored under `results/` (of csv format). A printout will be provided after the evaluations are complete. It does not matter if the evaluation script terminates unexpectedly: simply rerun it and the results that are already obtained will not be repeated again. See [here](../README-en.md#disc-law-eval-benchmark) for detailed evaluation methods. See [ml3m documentation](https://charlie-xiao.github.io/ml3m/) to better understand the evaluation scripts.

- [Evaluation results](./stats/): the evaluations we have done in our technical reports. All statistics for [objective evaluation](./stats/objective/) are released, but only a partial example for [subjective evaluation](./stats/subjective/) is released.
