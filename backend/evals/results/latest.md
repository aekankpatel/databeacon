# DataBeacon evaluation - 2026-11-06T05:34:38+00:00

Mode: `live` · Provider: `groq` · Model: `openai/gpt-oss-20b` · Budgets: `{'max_iterations': 2, 'max_tool_calls': 24, 'max_steps_per_iteration': 6}`

## Aggregate

| Metric | Value |
|---|---|
| cases | 6 |
| task success rate | 0.833 |
| sql validity | 1.0 |
| sql gold match rate | 0.833 |
| tool selection f1 | 0.767 |
| numerical accuracy | 0.667 |
| entity recall | 0.833 |
| hallucination rate | 0.019 |
| avg latency s | 164.075 |
| avg total tokens | 21304.667 |
| avg tool calls | 5.0 |
| error recovery rate | 0.6 |

## Per case

| Case | Success | Entities | SQL gold | Num. acc. | Tool F1 | Halluc. | Latency (s) | Tokens |
|---|---|---|---|---|---|---|---|---|
| revenue_decline_attribution | ✅ | 1.0 | True | 0.0 | 1.0 | 0.0 | 233.11 | 36487 |
| discount_shift | ✅ | 1.0 | True | 1.0 | 0.6 | 0.0 | 199.6 | 24809 |
| shipping_by_region | ✅ | 1.0 | True | 1.0 | 1.0 | 0.0 | 102.38 | 13990 |
| return_rate | ❌ | 0.0 | False | 0.0 | 0.0 | None | 50.98 | 0 |
| revenue_trend | ✅ | 1.0 | True | 1.0 | 1.0 | 0.0 | 179.69 | 22468 |
| data_quality | ✅ | 1.0 | True | 1.0 | 1.0 | 0.0952 | 218.69 | 30074 |
