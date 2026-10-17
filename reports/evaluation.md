# Evaluation: team vs single call

Provider: gemini gemini-3.5-flash-lite (run on 17 October 2026)

| Metric | Team (multi-agent) | Single call |
|---|---|---|
| Key-fact recall | 0.969 | 0.844 |
| Sentences with valid citations | 1.0 | 1.0 |
| Sentences judged supported | 0.958 | 0.964 |
| Unanswerable questions handled honestly | 2/2 | 2/2 |
| LLM calls per question | 6.1 | 1.0 |
| Tokens per question | 4180.6 | 1075.4 |
| Seconds per question | 9.63 | 1.21 |

## Per question

| Question | Type | Team recall | Single recall | Team supported | Single supported |
|---|---|---|---|---|---|
| q1_returns | multi-hop | 1.00 | 1.00 | 1.00 | 1.00 |
| q2_revenue | conflict | 0.75 | 0.75 | 1.00 | 1.00 |
| q3_service | multi-hop | 1.00 | 1.00 | 1.00 | 1.00 |
| q4_models | comparison | 1.00 | 1.00 | 1.00 | 1.00 |
| q5_cells | single | 1.00 | 1.00 | 1.00 | 1.00 |
| q6_cargo | multi-hop | 1.00 | 1.00 | 0.67 | 0.75 |
| q7_trail_cost | single | 1.00 | 1.00 | 1.00 | 1.00 |
| q8_europe | single | 1.00 | 0.00 | 1.00 | - |
| q9_share | unanswerable | True | True | - | - |
| q10_ceo_salary | unanswerable | True | True | - | - |

For unanswerable questions the recall columns show True if the system honestly said it could not answer.
The 'supported' metric is judged by the same model that wrote the answers, so treat it as indicative.

## Reading the results

- The whole recall gap comes from one question (q8): the single call searched once and missed the investor update; the Planner's sub-questions found it. The team's advantage here is retrieval, not writing.
- Citation validity, support and honesty on unanswerable questions were the same for both.
- The team cost about 6x the calls, 4x the tokens and 8x the time.
- 8 answerable questions is too few to call a 0.125 recall difference significant.
