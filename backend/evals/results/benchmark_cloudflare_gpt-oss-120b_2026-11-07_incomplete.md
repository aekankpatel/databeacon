> **Six-case Cloudflare benchmark, 2026-11-07, incomplete: 4 of 6 cases scored.**
> Model `@cf/openai/gpt-oss-120b` on Workers AI, no fallback, no paid inference. Agent budgets: 2 iterations x 6 tool turns.
>
> - The container restarted during case 5. Cases 1-4 had completed and are scored from their stored runs (`--reuse-run`), not re-run.
> - Case 5 (`revenue_trend`) was restarted and then failed: Cloudflare answered HTTP 429 / error 4006 "daily free allocation of 10,000 neurons used up". Our ledger had counted 3,547 neurons on 2026-11-07. Cloudflare's meter also included the previous evening's 6,815, so the allocation did not reset at 00:00 UTC as assumed. The guard now counts a rolling 24 hours.
> - Case 6 (`data_quality`) was never run.
> - Neurons spent on this benchmark: 3,547.3 (cap 7,000). That is 2,930.9 for the four scored cases, 525.4 on the interrupted case-5 attempt, and 91.1 on the failed retry.
> - Provider-quota failures stop the suite under the existing rule, so case 5 is reported as not completed, not as an agent failure.
> - `error recovery rate 0.0` means no tool errors occurred (0 of 0), not that recovery failed.

# DataBeacon evaluation - 2026-11-07T00:43:25+00:00

Mode: `live` · Provider: `cloudflare` · Model: `@cf/openai/gpt-oss-120b` · Budgets: `{'max_iterations': 2, 'max_tool_calls': 24, 'max_steps_per_iteration': 6}`

A case succeeds only if every check passes: primary conclusion (structured answer), period, metric, grouping, gold numbers in cited evidence, task alignment and hallucination <= 10%.

## Aggregate

| Metric | Value |
|---|---|
| cases | 4 |
| task success rate | 1.0 |
| primary conclusion rate | 1.0 |
| period correct rate | 1.0 |
| metric correct rate | 1.0 |
| grouping correct rate | 1.0 |
| numerical evidence rate | 1.0 |
| alignment rate | 1.0 |
| sql validity | 1.0 |
| tool selection f1 | 0.95 |
| hallucination rate | 0.0 |
| avg latency s | 69.468 |
| avg total tokens | 18691.0 |
| avg tool calls | 4.0 |
| error recovery rate | 0.0 |
| scorer calibration | 1.0 |
| total cloudflare neurons | 2930.9 |
| benchmark neurons incl interrupted work | 3547.3 |

## Per case

| Case | Expected | Success | Conclusion | Period | Metric | Grouping | Numbers | Aligned | Halluc. | Tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| revenue_decline_attribution | pass | ✅ | pass | pass | pass | pass | pass | pass | 0.0 | 20246 |
| discount_shift | pass | ✅ | pass | pass | pass | pass | pass | pass | 0.0 | 18251 |
| shipping_by_region | pass | ✅ | pass | n/a | pass | pass | pass | pass | 0.0 | 19616 |
| return_rate | pass | ✅ | pass | n/a | pass | pass | pass | pass | 0.0 | 16651 |
