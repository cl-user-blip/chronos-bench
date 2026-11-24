# 使用说明

## 环境与安装

Python 3.11 或更高版本。运行时依赖 NumPy、pandas、scikit-learn 和 threadpoolctl，不需要 GPU。

在仓库目录创建并激活虚拟环境，然后安装：

```bash
python -m pip install -e ".[dev]"
python -m chronos_bench --help
```

安装后也可以使用等价的 `chronos-bench` 控制台命令。`dev` 额外依赖用于运行 pytest。

## 命令

### 合成数据演示

```bash
python -m chronos_bench demo --rows 3360 --horizons 1 6 24 --seed 42 --alpha 0.1 --out runs/demo
```

演示数据包含日周期、周末效应、晚高峰、相关噪声和缓慢漂移，数值单位为任意单位。它用于验证评估流程，不用于估计实际负荷业务中的误差水平。

### CSV 评估

```bash
python -m chronos_bench evaluate --csv data/load.csv --horizons 1 6 24 --seed 42 --alpha 0.1 --out runs/load
```

每个预测距离分别拟合 Seasonal naive、Ridge lags、Ridge calendar、HistGB lags、HistGB calendar。模型与搜索空间固定在代码中，见 [实验设计](design.md)。CLI 不接收任意估计器或自定义搜索空间。

| 参数 | 适用命令 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `--csv PATH` | `evaluate` | 必填 | 输入 CSV 路径 |
| `--rows N` | `demo` | `3360` | 合成数据小时数，至少 720 |
| `--horizons H ...` | 两者 | `1 6 24` | 一个或多个互不重复的正整数预测距离，单位为小时 |
| `--alpha FLOAT` | 两者 | `0.1` | 区间显著性水平，必须介于 0 和 1；名义覆盖率为 `1-alpha` |
| `--seed INT` | 两者 | `42` | 合成数据和估计器使用的随机种子 |
| `--out PATH` | 两者 | `runs/demo` 或 `runs/csv` | 输出目录，自动创建 |

输出目录中的同名报告文件会被覆盖。比较不同实验时使用不同目录，例如 `runs/load-h6` 与 `runs/load-h24`。

## CSV 数据契约

```csv
timestamp,y
2025-01-01T00:00:00Z,110.5
2025-01-01T01:00:00Z,108.2
```

- 必须包含 `timestamp` 和 `y` 两列；额外列不参与评估。
- 至少 720 行。时间必须按小时等间隔、严格递增，不允许重复或缺口。
- `y` 必须能解析为有限数值；缺失、无穷值和无效时间戳会被拒绝。
- 时间转换为 UTC；没有时区的时间戳按 UTC 解释。小时、星期和周末特征使用转换后的时间。
- 程序不自动排序、插值、填充或归一化目标。输入预处理由调用方明确完成，并保持时间因果性。
- 720 行是基础最小值。每个块在按预测距离清除边界后仍需至少 24 个样本；较长的 `horizon` 或较小的 `alpha` 可能要求更多历史。

如果原始数据存在夏令时、非整点采样或缺测，应先确定重采样规则，再生成符合契约的输入。双向插值可能利用预测时点之后的观测，不宜直接作为通用预处理。

## 评估语义

时点 `t` 已知 `y[t]`，预测 `y[t+h]`。验证、校准与测试期间，真实观测逐小时到达，后续 origin 可以使用已经到达的观测；它不是从单一测试起点一次性预测整段未来。

原始观测按 55% 拟合、15% 验证、15% 校准、15% 测试划分。每块中的 origin 和 target 都必须位于相应时间边界内。候选超参数由验证集 MAE 选择，随后在前 70% 合法样本上重训。区间半宽仅由校准块残差确定，测试指标不参与模型或区间选择。

## 输出文件

| 文件 | 内容 |
| --- | --- |
| `report.json` | 配置、数据指纹、版本、时间块边界、全部模型结果与选择记录 |
| `predictions.csv` | 每个测试 origin、模型和 horizon 的逐点预测与区间 |
| `report.html` | 独立 HTML 报告，无 CDN 依赖，用本地浏览器打开 |
| `preview.svg` | 验证优胜配置的测试期预测图 |
| `input.csv` | 仅 `demo` 命令生成，用于复现合成序列 |

### `report.json`

- `schema_version`：当前为 1。
- `config`：预测距离、alpha 和 seed。
- `dataset.sha256`：经过校验、UTC 转换后，按 `pandas.to_csv(index=False)` 序列化得到的 UTF-8 字节摘要。它不是原始输入文件的字节哈希。
- `splits`：每个 horizon 的四段样本量与 origin/target 边界。
- `results`：每个模型与 horizon 的验证 MAE、测试 MAE/RMSE/WAPE、区间指标、候选搜索记录、重训和校准样本量。
- `selected_by_validation`：该 horizon 验证 MAE 最小的变体，不代表测试 MAE 最小。
- `refit_last_target`：重训样本的最后标签索引，应早于校准块首个 origin；季节基线无训练过程，值为 null。

WAPE 在测试目标绝对值之和为零时未定义，写为 JSON null。名义覆盖率不是时间相关、漂移数据上的分布无关保证；应同时检查经验覆盖率、平均宽度和 interval score。

### `predictions.csv`

字段为 `model,horizon,origin_index,target_index,target_timestamp,actual,prediction,lower,upper`。索引对应校验后的输入序列；`target_index = origin_index + horizon`。这些文件可能包含原始目标值，分享前需检查数据使用许可和访问范围。

## Python 接口

```python
from chronos_bench.benchmark import Config, run_benchmark
from chronos_bench.data import load_csv

frame = load_csv("data/load.csv")
report = run_benchmark(
    frame,
    "runs/api",
    Config(horizons=(1, 6, 24), alpha=0.1, seed=42),
)
```

`run_benchmark` 返回与 `report.json` 对应的字典，同时生成报告文件。接口使用相同的校验与时间切分逻辑，不会跳过 CLI 的实验约束。

## 退出码与常见错误

成功返回 0；CLI 参数、输入校验和常规文件访问错误返回 2，并在标准错误中说明原因。

| 错误 | 处理方式 |
| --- | --- |
| `Timestamps must be strictly increasing and unique` | 检查重复、乱序；在明确业务规则后预处理输入 |
| `This version requires hourly data without gaps` | 明确重采样和缺测处理规则，生成连续小时序列 |
| `fewer than 24 examples after horizon purge` | 增加数据量或缩短预测距离 |
| `Too few calibration examples for this alpha` | 增加校准样本量或提高 alpha |
| `SHA256 mismatch` | ETTh1 下载文件与固定版本不一致；不要跳过校验 |

固定版本 ETTh1 下载、来源与许可说明见 [ETTh1 实验](etth1.md)。
