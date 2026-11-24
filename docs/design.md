# 实验协议与可追溯性

## 预测任务

单变量、等间隔小时级序列。预测时点 `t` 已知 `y[0:t+1]`，直接预测 `y[t+h]`。每个 horizon 独立训练一个回归器，不把一步模型递归滚动 `h` 次。

报告采用 rolling-origin 的观测语义：随着时间推进，可以使用已经到达的真实观测作为下一时点的输入。它不是在测试起点一次性预测整个测试期。如果场景拿不到中间观测，需要另写 fixed-origin 协议。

## 边界先于模型

以原始观测行数切成 55% / 15% / 15% / 15% 四个连续块。拟合、验证、校准和测试的样本必须满足：

```text
block_start <= origin
target = origin + horizon < block_end
max(feature_source_index) <= origin
```

所以每块尾部约 `h` 个 origin 会被清除；不能仅按特征行划分数据，否则训练标签可能越过验证边界。验证和测试特征允许引用前一块的历史观测，因为上线时这些信息已经可用。

1. 在最早 55% 上拟合候选模型，StandardScaler 也只拟合该块。
2. 以随后 15% 的 MAE 选择每个模型变体的超参数；同时标记各 horizon 的验证优胜变体。
3. 固定超参数，在最早 70% 的所有合法 origin 上重新拟合，仍清除跨越 70% 边界的标签。
4. 在独立的 15% 校准块上计算绝对残差与区间半宽。
5. 在最后 15% 报告指标，测试结果不能反向影响前四步。

HistGradientBoosting 的 `early_stopping=False`，避免隐式内部随机 holdout。候选搜索空间固定且体积小；它强调协议审计，不能替代业务数据上的充分调优。

## 特征与基线

`lag_0,1,2,23,24,47,167` 加上 6/24/168 小时右对齐均值与标准差。`lag_0` 是预测时刻已经观测到的当前值，不是未来标签。日历变体增加目标时刻已知的小时/星期 sin-cos 和周末指示器；没有使用未来天气等不可获得信息。

| 变体 | 参数候选 | 用途 |
|---|---|---|
| Seasonal naive | 周期 24；使用最新已知同小时观测 | 不训练的强周期基线 |
| Ridge lags / calendar | alpha 0.1 / 10 / 100 | 线性规律、日历消融 |
| HistGB lags / calendar | max_leaf_nodes 7 / 15；80 轮 | 非线性规律、日历消融 |

`h > 24` 时季节基线使用 `t+h-ceil(h/24)*24`，不会错误读取 `t` 之后的同日观测。

## 区间与指标

令校准绝对残差为 `r_i`，样本数为 `n`，取排序后第 `ceil((n+1)*(1-alpha))` 个值为 `q`，预测区间是 `[prediction-q, prediction+q]`。样本不足以得到有限半宽时明确报错。报告 MAE、RMSE、WAPE、区间覆盖率、平均区间宽度以及 interval score。全零目标的 WAPE 未定义，写为 JSON null。

有限样本秩修正来自 split conformal 的方法，但此处时间序列有自相关、漂移，不满足可交换性；**不承诺分布无关的 90% 覆盖保证**。这个版本故意保留实际覆盖偏差，便于讨论滚动校准、按小时分桶和自适应 conformal 的后续方案。

## 数据契约

CSV 必须有 `timestamp,y` 两列，至少 720 行，时间严格递增、无重复、无缺口，目标全部有限。接受可解析时间戳并转 UTC。无时区时间按 UTC 解释；业务方应先显式转换本地时间。小时特征取 UTC 小时，因此真实本地负荷业务需要在特征层加入业务时区。当前拒绝缺失值，不自动插值、排序或前向填充。避免预处理悄悄改写预测问题。

合成数据用固定种子的日/周周期、相关噪声、晚高峰和缓慢漂移构造，只用于功能演示，不冒充真实数据集。

## 可追溯输出

- `report.json`：配置、规范化 frame 序列化为 CSV 后的 SHA256、运行环境、各块行边界、每个候选验证分数和最终指标。这个指纹不是下载文件的字节级哈希；真实数据下载器另外记录上游文件的哈希。
- `predictions.csv`：每个模型每个 horizon 的 origin/target 索引、目标时间、实际值、预测值和上下界。
- `report.html`：无 CDN、无服务端依赖的分享页。
- `preview.svg`：直接用于 README 的测试期预测图，来自验证优胜配置。

模型和数据处理以单线程执行，固定随机种子；跨版本线性代数实现可能带来小数差异。时间戳和运行耗时不是可复现性断言的一部分。

## 参考依据

实现为独立编写，以下官方文档用于核对时间拆分和估计器 API：

- [scikit-learn：带滞后特征的时间序列预测](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html)
- [scikit-learn：TimeSeriesSplit 的 gap 参数](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)；本项目使用显式 origin/target 边界而不是直接调用该类。
- [scikit-learn：HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
- [Angelopoulos & Bates：A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511)：区间秩修正与假设的参考；本项目不将可交换数据上的保证扩展到漂移时间序列。

扩展方向：真实公开数据的版本锁定与许可、扩展窗多折验证、按 horizon/hour 校准、分位数模型、概念漂移检测、模型序列化与服务化。当前只完成单序列离线实验平台。
