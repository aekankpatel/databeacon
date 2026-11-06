> Targeted live retest of one case after the task-alignment and token changes (Cloudflare Workers AI, `@cf/openai/gpt-oss-120b`, 1,066 neurons). Not the full benchmark.

# DataBeacon evaluation - 2026-11-06T23:20:54+00:00

Mode: `live` · Provider: `cloudflare` · Model: `@cf/openai/gpt-oss-120b` · Budgets: `{'max_iterations': 2, 'max_tool_calls': 24, 'max_steps_per_iteration': 6}`

A case succeeds only if every check passes: primary conclusion (structured answer), period, metric, grouping, gold numbers in cited evidence, task alignment and hallucination <= 10%.

## Aggregate

| Metric | Value |
|---|---|
| cases | 1 |
| task success rate | 1.0 |
| primary conclusion rate | 1.0 |
| period correct rate | 1.0 |
| metric correct rate | 1.0 |
| grouping correct rate | 1.0 |
| numerical evidence rate | 1.0 |
| alignment rate | 1.0 |
| sql validity | 1.0 |
| tool selection f1 | 0.857 |
| hallucination rate | 0.0 |
| avg latency s | 63.98 |
| avg total tokens | 27268.0 |
| avg tool calls | 8.0 |
| error recovery rate | 1.0 |
| scorer calibration | 1.0 |
| total cloudflare neurons | 1065.8 |

## Per case

| Case | Expected | Success | Conclusion | Period | Metric | Grouping | Numbers | Aligned | Halluc. | Tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| revenue_decline_attribution | pass | ✅ | pass | pass | pass | pass | pass | pass | 0.0 | 27268 |
