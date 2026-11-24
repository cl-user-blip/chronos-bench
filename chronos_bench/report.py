"""Dependency-free HTML/SVG reporting: works locally with no CDN or server."""

import html
from pathlib import Path

import numpy as np


def forecast_svg(sample, title: str) -> str:
    width, height = 1040, 360
    left, right, top, bottom = 70, 24, 52, 50
    low = float(sample[["actual", "lower"]].min().min())
    high = float(sample[["actual", "upper"]].max().max())
    span = max(high - low, 1)
    low, high = low - span * 0.08, high + span * 0.08
    xx = np.linspace(left, width - right, len(sample))
    def coords(values):
        yy = height - bottom - (np.asarray(values) - low) / (high - low) * (height - top - bottom)
        return list(zip(xx, yy))
    def points(values):
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in coords(values))
    band = coords(sample.upper) + coords(sample.lower)[::-1]
    polygon = " ".join(f"{x:.1f},{y:.1f}" for x, y in band)
    grid = ""
    for value in np.linspace(low, high, 5):
        ypos = height - bottom - (value - low) / (high - low) * (height - top - bottom)
        grid += f'<line x1="{left}" y1="{ypos:.1f}" x2="{width-right}" y2="{ypos:.1f}" stroke="#25344d"/><text x="{left-12}" y="{ypos+4:.1f}" text-anchor="end" fill="#9cabc4" font-size="12">{value:.0f}</text>'
    first = html.escape(str(sample.target_timestamp.iloc[0])[:16])
    last = html.escape(str(sample.target_timestamp.iloc[-1])[:16])
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Forecast with empirical calibrated interval">
<rect width="1040" height="360" rx="18" fill="#111d31"/>
<g font-family="Arial, sans-serif"><text x="70" y="30" fill="#eaf2ff" font-size="16" font-weight="bold">{html.escape(title)}</text>{grid}
<polygon points="{polygon}" fill="#38bdf8" fill-opacity="0.14"/>
<polyline points="{points(sample.actual)}" fill="none" stroke="#edf2fa" stroke-width="2"/>
<polyline points="{points(sample.prediction)}" fill="none" stroke="#38bdf8" stroke-width="2"/>
<text x="70" y="339" fill="#9cabc4" font-size="12">{first} UTC</text><text x="1016" y="339" text-anchor="end" fill="#9cabc4" font-size="12">{last} UTC</text>
<text x="625" y="30" fill="#edf2fa" font-size="12">Actual</text><text x="702" y="30" fill="#38bdf8" font-size="12">Prediction + interval</text>
<text x="18" y="190" transform="rotate(-90 18 190)" fill="#9cabc4" font-size="12">Target / original units</text></g></svg>'''


def render_report(report: dict, predictions, path: Path) -> str:
    first_horizon = report["config"]["horizons"][0]
    winner = next(r for r in report["results"] if r["horizon"] == first_horizon and r["selected_by_validation"])
    sample = predictions[(predictions.horizon == first_horizon) & (predictions.model == winner["model"])].head(168)
    svg = forecast_svg(sample, f"{winner['model']} · horizon {first_horizon}h · first test week")
    rows = ""
    for row in report["results"]:
        label = row["model"] + (" ★" if row["selected_by_validation"] else "")
        wape = "n/a" if row["wape"] is None else f"{row['wape']:.2%}"
        rows += f"<tr><td>{row['horizon']}h</td><td>{label}</td><td>{row['validation_mae']:.3f}</td><td>{row['mae']:.3f}</td><td>{row['rmse']:.3f}</td><td>{wape}</td><td>{row['coverage']:.1%}</td><td>{row['mean_width']:.2f}</td></tr>"
    source = html.escape(report["dataset"]["source"])
    text = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chronos Bench · Experiment report</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#091322;color:#eaf2ff;font:15px/1.7 system-ui,sans-serif}}main{{max-width:1160px;padding:52px 28px;margin:auto}}.tag{{color:#38bdf8;letter-spacing:3px;font-size:12px}}h1{{font-size:46px;line-height:1.2;margin:15px 0}}h2{{font-size:23px}}p{{color:#a6b5ce}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:30px 0}}.card{{padding:20px;border:1px solid #25344d;background:#111d31;border-radius:14px}}.value{{font-size:25px;font-weight:700}}.caption{{color:#9cabc4;font-size:12px}}svg{{width:100%;height:auto}}.table{{overflow:auto;border:1px solid #25344d;border-radius:14px}}table{{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}}th,td{{padding:13px 16px;border-bottom:1px solid #25344d;text-align:right}}th{{color:#9cabc4;background:#111d31}}td:nth-child(2),th:nth-child(2){{text-align:left}}.note{{padding:20px;border-left:3px solid #38bdf8;background:#111d31;border-radius:8px;color:#a6b5ce}}code{{color:#77ddca}}footer{{margin-top:30px;color:#9cabc4;font-size:12px}}@media(max-width:700px){{.cards{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:34px}}main{{padding:30px 16px}}}}
</style><main><div class="tag">CHRONOS BENCH / EXPERIMENT 001</div><h1>Forecasts you can audit.</h1><p>多步时序预测 · 时间边界审计 · 独立区间校准 · 原始结果可追溯</p>
<div class="cards"><div class="card"><div class="caption">DATA SOURCE</div><div class="value">{'Synthetic' if 'synthetic' in source else 'CSV'}</div><div class="caption">{report['dataset']['rows']:,} hourly observations</div></div>
<div class="card"><div class="caption">FORECAST HORIZONS</div><div class="value">{' / '.join(map(str,report['config']['horizons']))}h</div><div class="caption">Direct models per horizon</div></div>
<div class="card"><div class="caption">COMPARISONS</div><div class="value">5 variants</div><div class="caption">Seasonal / Ridge / HistGB</div></div>
<div class="card"><div class="caption">INTERVAL TARGET</div><div class="value">{1-report['config']['alpha']:.0%}</div><div class="caption">Empirical, no time-series guarantee</div></div></div>
{svg}<h2>Test results & feature ablation</h2><p>★ 在验证集上选出的配置；所有超参数以 validation MAE 选择，测试数据仅用于最终报告。表内为 refit 后的测试指标。</p>
<div class="table"><table><thead><tr><th>Horizon</th><th>Model / feature set</th><th>Val MAE</th><th>Test MAE</th><th>RMSE</th><th>WAPE</th><th>Coverage</th><th>Width</th></tr></thead><tbody>{rows}</tbody></table></div>
<h2>Protocol before performance</h2><div class="note">55% 拟合 → 15% 调参 → 前 70% 重训 → 15% 独立校准 → 15% 测试。样本 origin 与 target 必须同时处于指定块内，边界处按 horizon 清除交叉标签。滚动评估允许在每个时点使用已经到达的真实观测。</div>
<p>数据：<code>{source}</code>。合成样例使用任意负荷单位，不代表真实业务收益、公开数据集排名或生产效果。序列相关及漂移会影响覆盖率，区间覆盖以本次测试的经验结果为准。</p>
<footer>SHA256 {report['dataset']['sha256']}<br>Python {report['environment']['python']} · NumPy {report['environment']['numpy']} · scikit-learn {report['environment']['scikit_learn']}<br>原始数据：report.json / predictions.csv · Generated {report['created_at_utc']}</footer></main></html>'''
    path.write_text(text, encoding="utf-8")
    return svg
