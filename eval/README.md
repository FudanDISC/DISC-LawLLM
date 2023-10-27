# DISC-Law-Eval Benchmark

- 评测数据集：[客观评测集](./datasets/objective/)和[主观评测集](./datasets/subjective/)。

- [评测代码](./src/)：运行 `src/main.py` 进行评测。模型设定见 `src/models.py`（其中模型路径需自行修改）。主观评测的 few-shot 样例见[此目录](./src/few_shot/)。当运行评测代码时，模型回复将会被保存在 `responses/` 文件夹下（格式与对应的评测数据集相同），而评测结果将会保存在 `results/` 文件夹下（csv 格式）。评测代码完成后，评测结果将会被打印。如果评测代码未能正常完成，只需要重新运行直至其完成为止。已经得到的数据和评测结果不会被重复生成。您可以在[此处](../README-en.md#disc-law-eval-benchmark)查看关于我们使用的评测方法的更多详情。您也可以查看 [ml3m 技术文档](https://charlie-xiao.github.io/ml3m/)以便更好地理解我们的评测代码。

- [评测结果](./stats/): 此处保存了我们的技术报告中的评测结果。[主管评测](./stats/objective/)的所有数据均已发布，但我们只会发布部分[客观评测的样例](./stats/subjective/)。
