# DataBeacon evaluation - 2026-11-07T05:42:49+00:00

Mode: `scripted` · Provider: `scripted` · Model: `scripted` · Budgets: `{'max_iterations': 3, 'max_tool_calls': 24, 'max_steps_per_iteration': 8}`

A case succeeds only if every check passes: primary conclusion (structured answer), period, metric, grouping, gold numbers in cited evidence, task alignment and hallucination <= 10%.

## Aggregate

| Metric | Value |
|---|---|
| cases | 2 |
| task success rate | 1.0 |
| primary conclusion rate | 1.0 |
| period correct rate | 1.0 |
| metric correct rate | 1.0 |
| grouping correct rate | 1.0 |
| numerical evidence rate | 1.0 |
| alignment rate | 1.0 |
| sql validity | 0.584 |
| tool selection f1 | 1.0 |
| hallucination rate | 0.0 |
| avg latency s | 0.97 |
| avg total tokens | 25760.5 |
| avg tool calls | 8.5 |
| error recovery rate | 1.0 |
| scorer calibration | 1.0 |
| total cloudflare neurons | 0.0 |

## Per case

| Case | Expected | Success | Conclusion | Period | Metric | Grouping | Numbers | Aligned | Halluc. | Tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| revenue_decline_attribution | pass | ✅ | pass | pass | pass | pass | pass | pass | 0.0 | 20273 |
| calibration_wrong_quarter | fail | ❌ | fail | fail | pass | pass | fail | fail | 0.0 | 29443 |
| calibration_wrong_conclusion | fail | ❌ | fail | pass | pass | pass | pass | fail | 0.0 | 29722 |
| calibration_self_corrects | pass | ✅ | pass | pass | pass | pass | pass | pass | 0.0 | 31248 |
