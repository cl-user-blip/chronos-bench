# ETTh1 真实数据实验

## 来源、版本与使用范围

- 作者与原始仓库：[Haoyi Zhou et al. / ETDataset](https://github.com/zhouhaoyi/ETDataset)
- 固定 commit：`1d16c8f4f943005d613b5bc962e9eeb06058cf07`
- 原始文件：[ETT-small/ETTh1.csv](https://github.com/zhouhaoyi/ETDataset/blob/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTh1.csv)
- 原始文件 SHA256：`f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`
- 17,420 行，小时级；目标列 OT（oil temperature）。源仓库另有六个外生负荷变量，本项目未使用。
- 上游许可：[Creative Commons Attribution-NoDerivatives 4.0 International](https://github.com/zhouhaoyi/ETDataset/blob/1d16c8f4f943005d613b5bc962e9eeb06058cf07/LICENSE)。这不是 MIT 许可；本项目 MIT 仅适用于自己编写的代码和文档。

本项目不把原始数据或列提取后的本地文件重新分发。仓库只保留聚合评估指标及版本记录。下载器把上游原始 CSV、原始许可和转换后的 CSV 放进忽略目录，本地运行报告与逐点预测也在忽略目录内。准备发布时应继续排除 `data/` 与 `runs/`。

## 协议与复现

```bash
python scripts/fetch_etth1.py --out data/etth1
python -m chronos_bench evaluate --csv data/etth1/load.csv --horizons 1 6 24 --seed 42 --alpha 0.1 --out runs/etth1
```

下载器只请求上述固定版本的公开文件和 LICENSE，强制比对两者 SHA256，不使用会移动的 main URL。已有原始文件时也必须校验：

```bash
python scripts/fetch_etth1.py --source /path/to/ETTh1.csv --license-source /path/to/LICENSE --out data/etth1
```

本地转换仅保留 date、OT，分别改名 timestamp、y；不滤行、不对目标做归一化。上游时间戳没有时区，本框架保留时钟读数并解释为 UTC，此处不声称知道采集时区。

按原始序列行数切分 55% 拟合、15% 验证、15% 校准、15% 测试；边界按 horizon 清除；选参后重训最早 70%。这与 Informer 等文献的常用标准切分和多变量任务不一致，**不能作为同协议的论文横向排名结果**。

## 输出解释

[metrics.json](../examples/etth1/metrics.json) 包含五变体 × 三个 horizon 的全部聚合指标、候选搜索记录、数据块行边界、运行环境和来源 manifest，不包含逐点真实目标。

在这一协议中，24h 的验证优胜模型 Ridge lags 的区间经验覆盖率仅约 84.8%，说明不能把时间序列上的“名义 90%”直接表述为已达到保证。真实数据也揭示日历特征并非所有 horizon 都更好，因而保留所有消融结果。

## 原作者论文致谢

Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. **Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting.** AAAI 2021, 35(12), 11106–11115. [作者仓库中的引用信息](https://github.com/zhouhaoyi/ETDataset#electricity-transformer-dataset-etdataset)。

本项目提供数据契约、特征、拆分、实验调度和报告实现；与 ETDataset/Informer 官方项目无隶属关系。
