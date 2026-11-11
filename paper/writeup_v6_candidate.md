# Move On Until the Next Issue Is Worse

**Conditional restart yield, queue crossing, and runtime value for Gemma 4 coding agents**

## Abstract

Breadth-first scheduling repairs 43% more issues than immediate retries at a constrained call budget, yet selective retries improve on breadth-first itself. Repeated coding trajectories explain both results: a failed issue offers only 0.39–0.53 times the unconditional success probability across five model configurations. On 5,056 tasks, breadth-first adds 816 repairs at 30% of the observed three-attempt budget. A matched queue comparison then isolates 30.8 additional repairs from letting cheap retries overtake weak fresh issues. Gemma 4 wall-time replay reproduces the breadth advantage, reaching 70% at the same budget fraction. Its execution audit also reveals why counting accepted tools misses substantial computation: one attempt makes 314 model calls while registering 69 tools. Finally, a stopping-value identity predicts queue replay with R² = 0.996 across 105 settings. Selecting monitors by AUC instead of decision value forfeits 13.8 held-out repairs. These results connect repeated failure, scheduler design, and harness accounting through the opportunity cost of the next model call.

## 1. The queue changes the question

A coding agent can spend its next minute retrying a failed issue or opening another one. The competition makes that choice concrete: Gemma 4 serves a serial queue within a shared 12-hour session. Per-issue allowances do not guarantee that later issues receive compute. Twelve hour-long attempts can consume the session before the queue advances.

Repeated sampling measures whether another attempt can succeed. Scheduling additionally asks what that attempt displaces. We connect these questions through four empirical reversals. First, failure substantially lowers restart yield despite stochastic generation. Second, this favors breadth without making strict breadth optimal. Third, accepted tool counts can hide expensive failures before the scheduler sees them. Fourth, accurate failure classification need not select a valuable stopping rule.

Our evidence combines repeated outcomes, paired scheduling replay, and direct Gemma execution traces. Queue gains refer to reordering recorded attempts under matched budgets. The Gemma traces were collected on one A100 80GB through the competition harness. Their measured costs anchor wall-time replay; four-L4 deployment throughput remains a separate hardware quantity.

## 2. Repetition does not erase issue difficulty

Let p denote the success probability of an attempt drawn with equal issue weight. Let q denote another attempt's success probability after observing failure on that issue. With exchangeable binary attempts and intraclass correlation ρ,

$$q=p(1-ρ).$$

The relation follows from the covariance of two outcomes on one issue. Positive correlation concentrates repeated failures on persistent difficulties. Restarting therefore competes with a fresh draw at a reduced conditional probability, even when the generation seed changes.

Our primary dataset contains three seeded Qwen3.8-27B OpenHands rollouts on each of 5,056 SWE-rebench-V2 tasks. The public overlay covers 6,272 tasks; inclusion requires three valid scored rollouts. Among 1,216 excluded tasks, 1,191 have no valid rollout. Results therefore describe the complete-triplet population. Sampling uses temperature 1.0 and a one-hour timeout. Success rates by rollout position are 0.590, 0.601, and 0.591, supporting the exchangeability approximation.

Per-attempt success is 0.594, but conditional restart success is 0.273. Their ratio is 0.459, with a task-bootstrap 95% interval of 0.442–0.476. The ANOVA intraclass correlation is 0.541. Of all failures, 60.5% occur on tasks unsolved in every rollout. A beta-binomial fit predicts pass@3 of 0.757 against the observed 0.755.

Costs reinforce the probability gap. Successful attempts average 57.5 calls; failed attempts average 84.8. Fresh issues yield 8.67 repairs per 1,000 calls, while failure-conditioned restarts yield 3.43. The restart-to-fresh yield ratio is 0.396 by calls and 0.382 by wall time. Repeated failures consume more compute while offering fewer repairs.

**Table 1. Persistent issue difficulty reduces restart yield across model scales.** Ratios use equal issue weighting; intervals bootstrap issues. Gemma values concern the reference-passing local sandbox subset.

| Model and scaffold | Issues | Conditional-to-fresh success ratio | 95% interval |
|---|---:|---:|---:|
| Qwen3.8-27B, OpenHands | 5,056 | 0.459 | 0.442–0.476 |
| Gemma 4 31B, competition harness | 76 | 0.440 | 0.307–0.590 |
| Llama 3 8B, SWE-agent | 526 | 0.49 | 0.40–0.58 |
| Llama 3 70B, SWE-agent | 3,504 | 0.39 | 0.37–0.42 |
| Llama 3 405B, SWE-agent | 100 | 0.53 | 0.44–0.62 |

The Llama trajectory collection exposes a sampling trap. Issues have up to 91 attempts, and easier issues receive more attempts. Pooling all attempts produces an 8B restart ratio of 1.05. Drawing exactly three attempts per issue restores equal weight and gives 0.49. Sampling frequency can reverse the apparent conclusion about whether failure predicts future success. The source's adaptive collection design also means these estimates describe its recorded attempt distribution.

Cross-system outcomes support persistent issue difficulty through a different comparison. Four systems share 465 SWE-bench Verified issues. Across twelve ordered system pairs, success after another system's failure averages 0.48 times the receiving system's base rate. Every pair's interval lies below one. We keep this cross-system statistic separate from within-model restart yield.

### Breadth captures the first opportunity

A retry-first queue gives an issue up to three attempts before advancing. Breadth-first tries every issue once, then revisits failures. We replay both over the same task resamples and rollout orders, using 1,000 paired replicates. Budgets are fractions of the total observed cost of all three rollouts. An attempt that exceeds the remaining budget contributes no repair, and the queue ends there.

At 30% of total calls, retry-first repairs about 1,887 issues and breadth-first repairs 2,704. The paired difference is 816, with a 95% interval of 723–917. Breadth gains about 43% near this constrained-budget regime. At sufficiently large budgets, both policies reach the same observed coverage. The benefit comes from reallocating scarce compute across issues, rather than changing any generated patch.

Gemma supplies a measured wall-time counterpart. Its 228 attempts comprise three seeds on 76 reference-passing issues. At 30% of their aggregate runtime, breadth-first yields 20.3 repairs against 11.9 for retry-first. The difference is 8.3, with an interval of 1–17, approximately a 70% increase. Gemma's conditional ratio is 0.440 and its intraclass correlation is 0.564.

## 3. Breadth wins until the next issue is worse

A pooled fresh-issue rate hides the queue's composition. The strongest fresh issues leave first when the queue is ranked. Eventually a cheap restart can offer more repairs per call than the remaining fresh issues. The relevant comparison is therefore conditional restart yield against the next available alternative.

For an available restart with success probability q and expected cost c, admit it ahead of a fresh issue when its estimated yield exceeds that fresh issue's yield. Our greedy scheduler applies this comparison to all available attempts. Successful issues leave the queue. A later attempt becomes available only after failure. This policy is a practical ordering rule for the observed finite queue; future outcomes and costs remain unknown when it chooses.

We fit fresh yield by language on 2,592 training issues. Restart estimates also condition on the failed attempt's call tercile and the number of prior failures. The held-out partition contains 2,464 issues. An issue-ID hash fixes membership. The analysis uses 250 paired issue-resampling and rollout-order replicates.

Three policies isolate the mechanism. Ranked breadth shares the fresh-issue ranking but enforces first attempts before retries and first retries before second retries. A deferred-rate control retains fresh-first ordering while ranking retries by yield. The adaptive scheduler also allows retries to cross the fresh queue. Comparing adaptive with control prices crossing; comparing control with ranked breadth prices retry ordering.

**Table 2. Queue crossing and retry ordering pay at different budget fractions.** All entries are held-out replay repairs. Bracketed ranges are paired 95% intervals; the final column compares adaptive with ranked breadth.

| Call budget | Ranked breadth | Deferred-rate control | Adaptive | Crossing gain | Total gain |
|---|---:|---:|---:|---:|---:|
| 30% | 1,418 | 1,418 | 1,449 | +30.8 [5.2, 56.8] | +30.8 [5.2, 56.8] |
| 40% | 1,642 | 1,642 | 1,661 | +18.6 [−1.8, 36.0] | +18.8 [−1.8, 36.0] |
| 50% | 1,752 | 1,802 | 1,805 | +2.6 [−4.8, 16.8] | +52.3 [28.0, 78.8] |

At 30%, about 280 early restarts solve 96 issues. They displace about 182 fresh attempts that would solve 65. Restart success per attempt is slightly lower, approximately 34% against 36%, but the restarts cost less. This is a cost advantage, not evidence that retries universally solve easier problems.

Crossing scarcely activates below 25% budget, while strong fresh issues remain. On a separate 5% budget grid, its gain peaks at 35%. By 50%, ordering retries contributes 49.7 repairs, accounting for nearly all the 52.3 total gain. The training partition places the crossing and retry-order peaks at 35% and 50%, respectively. A pilot can therefore identify a useful switching window using its own outcomes.

The split predates this scheduler comparison and was previously used for restart analysis. Estimates remain training-only, while the reported comparison is a subsequent analysis on an existing holdout. An independent collection tests a stronger claim than this within-dataset comparison.

<!-- OPENHANDS_CONFIRMATION_START -->
Independent OpenHands queue confirmation: NaN（未重算）. The completed result artifact is pending; no confirmation estimate is included here.
<!-- OPENHANDS_CONFIRMATION_END -->

A separate Llama 70B replay uses message counts as its cost proxy. Its peak occurs at 35% budget, with +10.9 repairs and an interval of 2.7–20.0. At the prespecified 30%, 40%, and 50% budgets, paired intervals include both negative and positive changes. This comparison concerns message-budget scheduling; direct call costs support the primary table.

<!-- FIGURE_1_INSERTION: figures/fig1_schedule_and_persistence.png -->

*Figure 1. **Cheap restarts overtake fresh issues near the end of the first pass.** Paired gains separate crossing from retry ordering across budget fractions. The 27B panel uses calls; the 70B panel uses messages. Bands show 95% intervals from the grid replay.*

## 4. A tool budget can hide an expensive failure

The scheduling unit must reflect actual resource consumption. We inspected Gemma 4 31B IT QAT w4a16 traces served by vLLM 0.19.1 on an A100 80GB. Runs used the competition tools, 32K context, temperature 1.0, top-k 64, a ten-minute cap, and 100 counted tool calls.

Before measuring model outcomes, we applied reference patches to all 129 public issues. The audit environment combined the vLLM image's Python 3.12 runtime, competition wheelhouse, and sandbox base test tools. Reference patches passed on 76 issues and failed on 53. Missing test dependencies included `inline_snapshot` and `dirty_equals`. This audit characterizes that reconstructed execution environment. It establishes which issues supported our repeated-rollout measurement, without assigning a ceiling to the official evaluation service.

Reference failures and model failures require different remedies. A missing test dependency changes whether a correct patch can be verified. Repeating generation cannot repair that measurement contract. Separating the reference audit from model outcomes preserves a clear denominator for conditional success and prevents infrastructure failures from being interpreted as persistent model difficulty.

Within the 228 Gemma attempts, 160 failed. Among these, 76 attempts across 48 issues encountered editing errors, and 38 across 28 issues encountered editing schema errors. These categories overlap. A rejected call can consume generation and wall time while leaving the accepted-tool counter unchanged.

The third attempt on `rich_2943` provides a direct accounting example. Its run summary records 314 model calls, 238 schema errors, 69 counted tool calls, and 604.9 seconds elapsed. The accepted-tool count remains below 100 when the wall-time limit terminates execution. The trace parser recovers 311 calls; the 314 figure is the run-summary counter. Preserving both counters makes their provenance explicit.

This mechanism motivates separate accounting for model requests, accepted tools, and rejected tool requests. A runtime gate can observe repeated schema rejection before the nominal tool allowance expires. The present evidence establishes the failure mechanism and its cost. Any repair gain from a new schema gate requires its own evaluated policy comparison.

A generic loop flag is not equivalent to schema rejection. Fifteen failed attempts were flagged as looping across fourteen issues. Another rollout then succeeds with probability 0.30, against 0.114 after other failures. Prioritizing those loops in wall-time replay adds 0.58 repairs at 40% budget, with an interval of −2 to 3. Failure taxonomy can inform a gate, while decision value determines whether deploying that gate pays.

## 5. A better classifier can choose a worse stop

Suppose full trajectories have total cost C and total repairs T. Their average yield is λ=T/C. A stopping rule saves R calls and sacrifices Y repairs that its stopped trajectories would eventually produce. Under a fluid queue approximation, the change at budget B is

$$ΔG=\frac{B(λR-Y)}{C-R}.$$

The rule helps exactly when sacrificed repairs per saved call are below the queue yield. This prices opportunity cost directly. AUC ranks success and failure across thresholds; it does not encode how much computation a stop saves. Two monitors with similar discrimination can sacrifice very different repair counts for the same saved budget.

We evaluate outgoing-action features at 5, 10, and 15 actions. Features describe named paths, source and test paths, repetition, focus, and early edits. Tool outputs, reference patches, tests, and final patches are excluded. Verified issues split into 280 training and 185 held-out cases. Training folds select the checkpoint, feature family, and bottom-20% threshold.

The value identity is evaluated over 60 Verified candidates, 30 Lite settings, and 15 Multilingual settings. Across these 105 settings, it predicts paired queue replay with R² = 0.996, slope 1.014, and mean absolute error 0.148 repairs. Its predicted sign agrees in 97% of settings. These are checks of the fluid approximation against finite recorded queues, using each evaluated stop set's outcomes and costs.

Selecting by training-fold AUC yields a summed held-out change of −1.22 repairs across four systems. Selecting by training replay value yields +12.57. The difference is 13.79, rounded to 13.8 repairs. Within-system Spearman correlations between held-out AUC and gain average 0.438. That correlation is distinct from the monitors' AUC values.

The selected OpenHands gate illustrates the mechanism. Its held-out AUC is 0.493, yet it adds 3.31 repairs. It stops 40 trajectories, sacrifices 20 eventual successes, and frees 1,164 calls. Sacrificed repairs per saved call are 0.0172, below queue yield 0.0218. Saving expensive computation makes the decision valuable despite near-chance ranking accuracy.

Transfer changes the break-even point. A Mini gate applied to 300 Multilingual runs sacrifices 18 successes while saving 911 calls. Its loss rate exceeds the target queue yield. The identity predicts −1.97 repairs; replay gives −1.94. Target-distribution calibration therefore belongs inside the deployment procedure.

<!-- FIGURE_2_INSERTION: figures/fig2_value_identity_and_gemma_audit.png -->
*Figure 2. **Resource accounting explains both hidden execution cost and stopping value.** The reference audit separates 76 passing issues from 53 failing cases. Gemma traces expose schema rejection outside accepted-tool counts. Across 105 settings, the value identity tracks replayed gain with R² = 0.996.*

## 6. Reproduction and use

The [reproduction notebook](reproduce.ipynb) and [analysis guide](../analysis/README.md) connect per-attempt records to restart estimates and paired queue comparisons. Result files retain budgets, populations, cost units, and intervals. The reference-patch audit and Gemma traces provide complementary evidence about evaluation validity and resource accounting.

[SWE-bench](https://arxiv.org/abs/2310.06770), [SWE-agent](https://arxiv.org/abs/2405.15793), and [SWE-rebench](https://arxiv.org/abs/2602.23866) supply the underlying tasks and trajectory collections. [FailFast-RestartSmart](https://arxiv.org/abs/2608.03222) studies monitored restarts. Our comparison adds the next available issue as the competing use of saved compute.

The replay contract also fixes what the scheduler can learn. Repeated attempts are fresh seeded runs, so their costs do not incorporate reuse of a failed patch. Success removes an issue from further scheduling. Resampling keeps each issue's attempts together, preserving within-issue dependence. Budget fractions normalize each collection by its own observed rollout cost; they do not equate calls across models or hardware.

The primary split tests new issues drawn from the recorded language mixture. A claim about an unseen language or repository family requires holding that entire family out during fitting. The present queue comparison estimates performance within the observed task mixture. Likewise, the two directly audited repeated-run configurations use temperature 1.0. A deployment with different sampling or task composition should estimate conditional yield on its own pilot before choosing a switching window.

Apply the rule with a pilot drawn from the intended queue. Estimate yields in the resource unit that binds deployment. Rank available attempts, permit queue crossing when conditional yield warrants it, and price stopping by saved cost and sacrificed repairs. Keep schema rejection visible in that accounting. The resulting scheduler follows opportunity cost as the queue changes.
