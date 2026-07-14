# algorithms 目录说明

每个算法独立目录，**report 实现在各自目录内**。

```
algorithms/smoke/
  config.py      # 默认参数（label、冷却时间等）
  report.py      # SmokeReport
  algorithm.py   # 可选：AlgorithmSlot 包装

registry.py 通过 algorithm_registry.json 加载：
  "report": {
    "policy_type": "custom",
    "module": "algorithms.smoke.report",
    "class_name": "SmokeReport"
  }
```

## 对照表

| API 名 | Report 类 | 文件 |
|--------|-----------|------|
| Smoke | SmokeReport | `algorithms/smoke/report.py` |
| Flame | FlameReport | `algorithms/flame/report.py` |
| Helmet | HelmetReport | `algorithms/helmet/report.py` |
| person_detection | PersonReport | `algorithms/person/report.py` |
| Car_Detection | CarReport | `algorithms/car/report.py` |
| Bag_Detection | BagReport | `algorithms/bag/report.py` |

公共能力仍在根目录：

- `report.py` → `Policy` 基类、`Async_Alarm` 上报
- `registry.py` → `create_report_policy()`

## 新增算法

1. 复制 `algorithms/_template/` 为 `algorithms/xxx/`
2. 改 `config.py`、`report.py`
3. 在 `algorithm_registry.json` 增加条目（`module` / `class_name` 指向新 report）
