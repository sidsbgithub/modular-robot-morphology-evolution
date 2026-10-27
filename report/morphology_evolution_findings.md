# Assignment 1 — Consolidated findings report (working document)

Evolutionary Computing 2026 · Standard Assignment 1 · Tree-genome robot morphology
Status: **draft of all findings up to 2026-09-20** — input for the 3-page GECCO report, not the report itself.

---

## 0. One-paragraph summary

We evolve robot bodies (Ariel `TreeGenome`) to match five target bodies; fitness is the mean tree-edit distance to the targets plus one standard deviation across targets (**lower is better**, 0 = identical). Across ~4,800 EA runs in four escalating stages (coarse one-factor screens → refined screens → a 420-configuration mutation×crossover factorial with a two-stage racing design → a held-out final experiment), the strongest and cleanest effect is the **mutation operator**: operators that *add* structure (subtree replacement) beat operators that only *remove* structure (hoist, shrink) by ~0.5–2.0 fitness units, and the mechanism is visible in the data — destructive operators drive the population to small trees (≈9–12 modules) while the targets have 7–25 modules and the best compromise bodies have ≈13–15. Crossover is the second-strongest factor (monotonic: none ≫ 0.2 > … > 1.0) and *interacts* with the operator: it is nearly essential for relabel/removal operators (gain 1.3–1.9) but only mildly helpful when subtree mutation already injects new material (gain 0.25–0.4). Selection scheme (truncation vs k-tournament vs linear ranking) had the weakest, non-monotonic effect and was dropped. Mutation rate plateaus from 0.5 upward; applying 2 mutation events instead of 1 makes no difference. The tuned baseline is `subtree, rate 0.75, 1 event, subtree crossover p=1.0, truncation (μ+λ)`. In the final held-out experiment (10 fresh seeds per variant, pop 60 × 100 generations, all five operators) the constructive operators **mixed 12.39 ± 0.35, subtree 12.49 ± 0.36 and replace 12.53 ± 0.46** were statistically indistinguishable from each other and significantly better than the destructive ones (**shrink 12.81 ± 0.46, hoist 12.99 ± 0.29**; subtree vs hoist p = 0.007, mixed vs hoist p = 0.001), and every EA run beat every run of the matched-budget random search (**17.02 ± 0.21**) — the EA passes random search's *final* value after a single generation (120 evaluations). Final population tree size follows the same order (13.9 → 12.8 modules). For scale: a random body scores ≈ 20.8 and even the target bodies themselves score 17.3–24.7 against the set, so ≈ 12.4 is a strong result on this problem.

---

## 1. What the assignment asks and how it is graded

| Requirement (from the PDF) | Status |
|---|---|
| Evolve a body whose phenotype resembles a target set; fitness = mean + std of weighted tree-edit distance, lower is better | Implemented (`calculate_fitness` → `mean_plus_std_tree_edit_distance`) |
| Investigate **one** aspect (mutation / crossover / selection) as the research question; the two variants differ in that one aspect only | Chosen: **mutation operator**; variants differ only in the operator (see §6) |
| ≥ 5 independent runs per variant, mean/spread statistics | Done: 10 held-out seeds per variant |
| Line-plot across generations, mean/std over runs | `fig_convergence.png` (produced by `A1_final_analysis.py`) |
| Random-search baseline at the same evaluation budget | Done: 10 seeds × 6,060 evaluations → 17.02 ± 0.21 (every EA run beats every random run) |
| No changes to `src/ariel`; own EA design using Ariel operators as building blocks | Respected — all code lives in `assignments/assignment_1/` |
| Report ≤ 3 pages, GECCO19 template; grade is based on the report, not the raw number | To do (§9) |

Rubric mapping (30 pts): problem definition 3 · algorithm 5 · methodology 5 · results & discussion 10 · conclusion 5 · structure 2. Sections 6–8 below map the findings onto those items.

A clarification that came up: **the objective is to minimise fitness**. Whenever this document says a configuration is "better", it means its fitness is *lower*.

---

## 2. The EA (as implemented in `A1_evolution_tree.py`)

- **Representation.** Ariel `TreeGenome`: a rooted tree with a CORE root; nodes carry module type (BRICK/HINGE) and rotation; edges carry the attachment face. Random initialisation with `random_tree(max_modules=20)`, depth capped at 12.
- **Fitness.** Zhang–Shasha tree edit distance (delete 1, insert 1, relabel 0 / 0.5 rotation mismatch / 1 type mismatch) to each of 5 targets; score = mean + population std of the five distances. Invalid genomes get +∞.
- **Generational scheme.** (μ+λ) with μ = λ = population size. Parent selection picks ~50 % of the population (truncation, k-tournament, or linear ranking); each offspring = crossover(parent_a, parent_b) → mutation → validation; survivors = best μ of parents ∪ offspring (elitist). Evaluations per run = pop × (generations + 1).
- **Variation operators** (all Ariel built-ins, `ariel.ec.genotypes.tree.operators`):

| Operator | What it does | Effect on tree size |
|---|---|---|
| `replace` (point mutation) | Re-labels one node's type/rotation; children on faces the new type disallows are pruned | ≈ neutral |
| `subtree` (subtree replacement, Koza-style) | Deletes a subtree and grafts a **fresh random subtree of 1–3 nodes** | injects new material |
| `shrink` | Deletes a whole subtree and replaces it with **one** random node | strongly reducing |
| `hoist` | Deletes an internal node **and all its sibling branches**, reconnecting one surviving child to the grandparent | reducing, keeps one evolved branch |
| `mixed` (ours) | Each event picks replace/subtree/shrink/hoist with p = .4/.4/.1/.1 | — |
| `crossover_subtree` | Swaps random subtrees between two parents; child validated, falls back to parent copy | — |

  Knobs added by us: `mutation_rate` (probability an offspring is mutated at all), `mutation_events` (how many operator applications when it is), `crossover_rate` (probability crossover fires per mating), `tournament_k`, `rank_pressure`, optional `max_age` (unused in all reported results).
- **Budget.** Final experiments: pop 60, 100 generations → 6,060 evaluations per run (inside the assignment's suggested 50–100 pop × 100 generations).

---

## 3. Methodology: a four-stage funnel

All stages share the same code path and seeds are fixed per run, so any run is reproducible from its `config_label` + seed. Fitness evaluation is a pure-Python tree comparison (milliseconds), so the bottleneck was run count, not simulation; runs were parallelised across 10–11 worker processes.

| Stage | Purpose | Budget per run | Runs | Output folder |
|---|---|---|---|---|
| 1. Coarse OFAT screen | Decide which axis is worth a research question | pop 25 × 20 gen (525 evals), 3 seeds | 156 (mutation 40 configs, crossover 4, selection 8) | `__data__/A1_evolution_tree/screening/` (overwritten by stage 2) |
| 2. Refined OFAT screen | Confirm stage-1 trends with more budget, add tree-size tracking | pop 35 × 35 gen (1,260 evals), 5 seeds | 290 | `screening/` |
| 3. Full factorial + racing | Find the best *combination* and its statistical support | pop 60 × 100 gen (6,060 evals); 420 configs × 8 seeds, then top-20 × 56 extra seeds | 4,480 (0 failures) | `factorial_final/` |
| 4. Final experiment | Answer the research question on **held-out seeds** (never used in tuning) | pop 60 × 100 gen, 10 seeds per variant + random search | 20 + 10 | `final_experiment/`, `random_search/` |

OFAT = one factor at a time: while one axis is swept, the other two are pinned to a shared baseline (`mixed r=0.5 e=1`, `subtree crossover p=1.0`, `truncation`). The factorial removed that limitation for the two axes that mattered.

Statistics: per-configuration mean ± std over seeds; two-sided Mann–Whitney U for pairwise comparisons (small n, no normality assumption); Holm correction when many finalists are compared against the best; Spearman rank correlation for size-vs-fitness; 95 % t-intervals in `final_ranking.csv`.

---

## 4. Findings from the screening stages (1–2)

### 4.1 Stage 1 (pop 25 × 20 gen, 3 seeds) — which axis moves fitness?

| Axis | Best config | Worst config | Range of config means |
|---|---|---|---|
| Mutation | `mixed r=1 e=2` 13.57 (≈ `mixed r=0.5 e=1` 13.57) | `hoist r=1 e=2` 14.92 | **1.35** |
| Crossover | `subtree cxr=1` 13.57 | `none` 14.27 (monotonic: 1.0 → 0.6 → 0.3 → none = 13.57 → 14.00 → 14.18 → 14.27) | 0.70 |
| Selection | `truncation` 13.57 | `rank s=1.2` 14.13; tournament k = 2/3/5/7 → 13.91/13.72/13.98/13.71 (**non-monotonic**) | 0.56 |

Reading: mutation's range was dominated by hoist collapsing at high intensity; crossover showed a clean monotonic trend; selection was the weakest and noisiest (no trend in k) and was **dropped** from further study. (Sanity check that fell out of the design: the baseline configuration appears in all three screens and produced bit-identical means, 13.571047557065256, confirming full determinism of the pipeline.)

### 4.2 Stage 2 (pop 35 × 35 gen, 5 seeds) — trends hold, and the mechanism appears

- Mutation: `subtree r=0.5 e=1` 12.81, `mixed r=0.5 e=1` 12.82, `replace r=0.5 e=1` 12.88 at the top; `hoist r=1 e=2` 14.66 and `hoist r=0.75 e=2` 14.13 at the bottom. Within almost every operator, `e=1` beat `e=2`.
- Crossover: `cxr=1` 12.82 clearly best, `none` 13.66 clearly worst, intermediate rates bunched (13.27–13.59) with no clean order.
- **Tree size over generations** (population mean modules; all runs start at ≈21 because `random_tree` fills the 20-module budget):

| Generation | `subtree r=0.5 e=1` (fitness → 12.81) | `hoist r=0.5 e=1` (→ 13.38) | `hoist r=1 e=2` (→ 14.66) |
|---|---|---|---|
| 0 | 21.0 | 21.0 | 21.0 |
| 5 | 10.8 | 11.0 | 9.4 |
| 10 | 11.8 | 11.0 | 8.9 |
| 20 | 12.6 | 11.6 | 9.0 |
| 35 | 13.1 | 12.4 | 9.2 |

  Healthy configurations *contract then regrow*; the aggressive destructive configuration collapses to ≈9 modules by generation 5 and never recovers, and its fitness flat-lines from that point. This is the dose-dependent story: a little structural simplification is harmless (even useful), a lot — every offspring, twice — outruns the population's ability to rebuild.

---

## 5. Findings from the factorial + racing (stage 3, 4,480 runs)

### 5.1 The winner, and its statistical status

| Rank | Configuration | n | mean ± std | 95 % CI | Holm p vs #1 | Tied with #1? |
|---|---|---|---|---|---|---|
| 1 | `subtree r=0.75 e=1 \| cxr=1` | 64 | **12.34 ± 0.29** | 12.26–12.41 | — | — |
| 2 | `subtree r=0.75 e=1 \| cxr=0.4` | 64 | 12.38 ± 0.32 | 12.30–12.46 | 0.79 | yes |
| 3 | `subtree r=0.75 e=1 \| cxr=0.8` | 64 | 12.39 ± 0.35 | 12.30–12.48 | 0.79 | yes |
| 4 | `replace r=0.6 e=1 \| cxr=1` | 64 | 12.42 ± 0.31 | 12.34–12.50 | 0.71 | yes |
| 5 | `mixed r=1 e=1 \| cxr=0.8` | 64 | 12.42 ± 0.30 | 12.34–12.50 | 0.71 | yes |
| … | (13 of 19 finalists tied with #1 after Holm correction) | | | | | |
| 14 | `replace r=0.5 e=1 \| cxr=1` | 64 | 12.49 ± 0.32 | | 0.039 | no |
| 19 | `subtree r=0.75 e=1 \| cxr=0.2` | 64 | 12.63 ± 0.41 | | 0.0004 | no |
| 20 | `mixed r=0.4 e=2 \| cxr=0.8` | 64 | 12.67 ± 0.41 | | 0.0001 | no |

Interpretation: there is no single configuration that beats all others, and that is the honest answer. The *winning region* is well defined: a constructive operator (subtree, mixed, replace), mutation rate ≥ 0.5, crossover probability ≥ 0.4; events 1 vs 2 irrelevant. Configurations that stepped outside it (`cxr=0.2`, `r=0.4 e=2`) were significantly worse.

The racing stage earned its cost: the **stage-1 leader** (`mixed r=0.4 e=2 | cxr=0.8`, 12.23 on 8 seeds) finished **last of the 20 finalists** (12.67 on 64 seeds). A single-stage grid would have declared the wrong winner — a textbook winner's-curse / regression-to-the-mean effect worth one sentence in the paper's methodology.

### 5.2 Main effects (balanced design: 420 configs × 8 seeds = 3,360 runs)

| Factor | Level means (lower is better) | Significance (pairwise Mann–Whitney, adjacent levels) |
|---|---|---|
| **Operator** | subtree 12.67 · mixed 12.72 · replace 12.91 · shrink 13.29 · hoist 13.63 | all adjacent steps p < 0.02 (subtree vs mixed the weakest, p = 0.013, Δ = 0.05) |
| **Crossover** | 1.0 12.69 · 0.8 12.75 · 0.6 12.89 · 0.4 13.05 · 0.2 13.14 · none 13.73 | strictly monotonic; every step significant; none → 0.2 is the big jump (Δ 0.58, p = 3·10⁻²⁴) |
| **Mutation rate** | 0.1 13.27 · 0.25 13.10 · 0.4 13.03 · 0.5 12.97 · 0.6 13.01 · 0.75 12.91 · 1.0 13.00 | low rates significantly worse (0.25 vs 0.1 p = 10⁻⁵); **plateau from 0.5** (0.75 vs 0.5 p = 0.18; 0.75 vs 1.0 n.s.) |
| **Mutation events** | 1: 13.03 · 2: 13.06 | not significant (p = 0.13) — drop this knob |

Effect sizes (range of level means): operator ≈ 1.0, crossover ≈ 1.0, rate 0.36, events 0.03.

### 5.3 The operator × crossover interaction

Mean best fitness, and the gain from adding full crossover to a run with none:

| Operator | none | cxr 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | gain none → 1.0 |
|---|---|---|---|---|---|---|---|
| hoist | 14.91 | 13.79 | 13.55 | 13.38 | 13.11 | 13.03 | **1.88** |
| replace | 13.84 | 12.91 | 12.88 | 12.70 | 12.60 | 12.51 | **1.34** |
| shrink | 14.18 | 13.40 | 13.27 | 13.09 | 12.92 | 12.87 | **1.30** |
| subtree | 12.85 | 12.77 | 12.80 | 12.63 | 12.52 | 12.44 | 0.41 |
| mixed | 12.86 | 12.85 | 12.74 | 12.65 | 12.60 | 12.61 | 0.25 |

All five gains are significant (p ≤ 8·10⁻⁵), but they differ by a factor of 7. Mechanism: subtree replacement grafts fresh random structure into a body — functionally what crossover also does — so it partially *substitutes* for recombination. Relabelling (`replace`) and removal (`shrink`, `hoist`) cannot create structure, so those runs depend on crossover for all their exploration; without it, `hoist | none` fills the bottom ten of the entire grid (14.8–15.2). This also explains why the stage-1/2 OFAT crossover screens *understated* crossover: they were pinned to `mixed`, where it barely matters.

Operator × rate: hoist is the only operator that gets *worse* at rate 1.0 (14.00 vs ≈13.5 at lower rates); subtree is best at 0.75 (12.45).

### 5.4 Mechanism: tree size explains fitness

- Population mean modules at generation 100, by operator: hoist 12.1 · shrink 12.5 · replace 12.9 · mixed 13.1 · subtree 13.3. By crossover: none 11.6 · 0.2 12.6 · 0.4 12.9 · 0.6 13.1 · 0.8 13.3 · 1.0 13.3. Extremes: `hoist | none` 10.1, `subtree | cxr=1` 13.7.
- Across the 420 configurations, **Spearman ρ(final mean modules, fitness) = −0.72** (p ≈ 10⁻⁶⁸): configurations that end with larger trees score better.
- The targets have **7, 11, 15, 19 and 25 modules** (depths 4, 6, 4, 8, 7). Because the score is mean + std over targets, the optimum is a compromise body of roughly 13–15 modules; a 9–11-module body is far from three of the five targets. Operators that only remove structure therefore fight the fitness function directly. (Note the 20-module initialisation cap also means the 25-module target can never be matched exactly — a fixed floor in every fitness value.)

---

## 6. Final experiment (stage 4): constructive vs destructive mutation on held-out seeds

**Research question.** Do mutation operators that can *add* structure (subtree replacement, and the composite `mixed` that contains it; plus the size-neutral relabelling operator `replace`) produce better-matching morphologies than operators that only *remove* structure (`shrink`, `hoist`), all else equal — and is the difference explained by the size of the evolved bodies?

**Hypothesis.** Structure-removing operators reach worse (higher) fitness because they drive the population toward small bodies that cannot approximate the larger targets; structure-adding operators keep the population near the compromise size the fitness function rewards.

**Design.** Five variants identical in every respect except the mutation operator (`subtree`, `mixed`, `replace`, `shrink`, `hoist`): `mutation_rate 0.75, mutation_events 1, subtree crossover p = 1.0, truncation parent selection, (μ+λ) elitist survival, pop 60, 100 generations (6,060 evaluations), max_modules 20, max_depth 12`. Seeds 7001–7010 for every variant and for the random-search baseline (6,060 random bodies per seed, best-so-far reported); **none of these seeds were used during tuning**, so the numbers are not inflated by the selection in stage 3.

**Scale of the fitness values** (so that "12" can be interpreted): a single random body scores ≈ 20.8; the best of 6,060 random bodies ≈ 17.0; each *target body itself*, scored against the target set, gets 17.3–24.7 because the targets are 8.5–22 edits apart from one another; a crude lower bound that counts only size mismatch for a ~16-module body is ≈ 8.6. The EA therefore finds compromise bodies that represent the target set better than any single target does.

**Result** (10 independent runs per group, held-out seeds):

| Group | mean ± std | 95 % CI | median | min | population mean modules at gen 100 |
|---|---|---|---|---|---|
| mixed mutation | **12.39 ± 0.35** | 12.14–12.64 | 12.40 | 11.90 | 13.4 |
| subtree mutation | **12.49 ± 0.36** | 12.23–12.75 | 12.36 | 11.94 | 13.9 |
| replace mutation | 12.53 ± 0.46 | 12.20–12.86 | 12.45 | 11.71 | 13.3 |
| shrink mutation | 12.81 ± 0.46 | 12.48–13.13 | 12.66 | 12.13 | 13.1 |
| hoist mutation | 12.99 ± 0.29 | 12.79–13.20 | 12.95 | 12.57 | 12.8 |
| random search (6,060 evaluations) | 17.02 ± 0.21 | 16.87–17.17 | 17.01 | 16.73 | — |

Pairwise two-sided Mann–Whitney U tests (Cliff's δ; negative = first group lower):

| Comparison | Δ mean | p | δ |
|---|---|---|---|
| subtree vs hoist | −0.50 | **0.007** | −0.72 |
| mixed vs hoist | −0.60 | **0.001** | −0.86 |
| replace vs hoist | −0.46 | **0.014** | −0.66 |
| subtree vs shrink | −0.32 | 0.064 | −0.50 |
| mixed vs shrink | −0.41 | 0.082 | −0.47 |
| replace vs shrink | −0.28 | 0.076 | −0.48 |
| subtree vs mixed | +0.10 | 0.60 | +0.15 |
| subtree vs replace | −0.04 | 0.47 | −0.20 |
| shrink vs hoist | −0.18 | 0.27 | −0.30 |
| any EA variant vs random search | −4.0 to −4.6 | **0.0002** | −1.00 (every EA run beats every random run) |

Reading: the three constructive operators form one statistically indistinguishable group and the two destructive operators another, with the constructive group 0.3–0.6 better; the difference is significant against hoist and marginal (p ≈ 0.06–0.08 at n = 10) against shrink. Final population tree size follows the same order (mixed/subtree/replace 13.3–13.9 modules, shrink 13.1, hoist 12.8), consistent with the size mechanism — although with full crossover the size gap is much smaller than in the factorial's no-crossover cells (§5.4): crossover partly compensates for a destructive mutation operator. All five EA variants pass the random search's *final* value after one generation (120 evaluations) and finish ≈ 4.5 units lower. The held-out means of the constructive operators (12.39–12.53) are slightly worse than the tuning-set estimate of the single best configuration (12.34) — the expected optimism gap; report both.

Figures produced by `A1_final_analysis.py` in `final_experiment/`: `fig_convergence.png` (mean ± std best fitness per generation, five variants + random search aligned by evaluation count), `fig_tree_size.png` (population mean modules per generation with the five target sizes as reference lines), `fig_final_boxplot.png` (per-run final fitness), plus `final_stats.csv` and `pairwise_tests.csv`.

---

## 7. Limitations and unexpected findings (for the Discussion section)

1. **Ties, not a single winner.** 13 of 19 finalists are statistically indistinguishable from the best even with 64 seeds each; the defensible claim is a *region* of good settings, not a point.
2. **Winner's curse observed.** The 8-seed leader ranked last among finalists after 64 seeds — direct evidence that single-stage tuning on few seeds overfits noise.
3. **OFAT understated crossover** because the pinned mutation operator (`mixed`) already supplies novel material; the factorial was necessary to see the interaction.
4. **Fitness floor.** `max_modules = 20` at initialisation plus the 25-module target means the theoretical minimum is > 0; the std term further penalises any body that matches one target closely. We did not vary the target set or cost weights (not our research question).
5. **Per-target distances not recorded.** Runs deleted their sqlite databases to save ~40 GB, so the best genotypes are not retrievable for a per-target breakdown; if wanted, re-run the two final variants with `--keep-db`.
6. **Small n for the final comparison** (10 per variant) — the effect (Δ ≈ 0.5, δ = 0.72) is large enough to be significant, but the confidence intervals are wide; more seeds are cheap (≈ 25 s per run) if the group wants tighter intervals.
7. **Unused mechanisms in the code**: linear-ranking selection, tournament size and the optional age-based turnover were implemented for screening and are switched off in all reported results. Either remove them before submission or state explicitly that they were not used (the rubric penalises algorithm detail that does not match the report).
8. **Tuning winner ≠ held-out winner.** The factorial's best single configuration used `subtree`; on the held-out seeds `mixed` finished marginally ahead of `subtree` (Δ 0.10, p = 0.60). The correct statement is that the three constructive operators are tied — do not claim subtree is "the best operator", claim the constructive-vs-destructive split.
9. **Crossover masks part of the effect.** With crossover at p = 1.0 the destructive operators still end at 12.8–13.1 modules (vs 10–11 without crossover in the factorial); the operator effect on held-out seeds (Δ ≈ 0.5) is therefore smaller than the factorial's no-crossover cells suggest (Δ ≈ 2). This is the interaction of §5.3 and belongs in the Discussion.

---

## 8. Reproducibility (exact commands, run from the repository root)

```bash
# Stage 1 – coarse one-factor screens (mutation, crossover, selection)
python assignments/assignment_1/A1_evolution_tree.py --screen --screen-axis all --screen-pop 25 --screen-generations 20 --screen-seeds 42 123 456 --workers 4

# Stage 3 – full mutation x crossover factorial with top-20 racing (≈4 h on 11 workers)
python assignments/assignment_1/A1_evolution_tree.py --screen --screen-axis factorial --screen-name factorial_final --screen-pop 60 --screen-generations 100 --screen-seeds 42 123 456 789 1011 2024 31415 27182 --screen-mutations replace subtree shrink hoist mixed --screen-rates 0.1 0.25 0.4 0.5 0.6 0.75 1.0 --screen-events 1 2 --screen-crossover-options none subtree --screen-crossover-rates 0.2 0.4 0.6 0.8 1.0 --screen-selection truncation --refine-top-k 20 --refine-seeds 5001 ... 5056 --workers 11

# Stage 4 – final experiment on held-out seeds + matched-budget random search
python assignments/assignment_1/A1_evolution_tree.py --screen --screen-axis mutation --screen-name final_experiment --screen-pop 60 --screen-generations 100 --screen-seeds 7001 7002 7003 7004 7005 7006 7007 7008 7009 7010 --screen-mutations subtree mixed replace shrink hoist --screen-rates 0.75 --screen-events 1 --screen-crossover subtree --screen-selection truncation --workers 10
python assignments/assignment_1/A1_evolution_tree.py --random-search --random-search-budget 6060 --random-search-seeds 7001 7002 7003 7004 7005 7006 7007 7008 7009 7010
python assignments/assignment_1/A1_final_analysis.py
```

Software: Python 3.12, Ariel fork from the course repository (unmodified `src/ariel`), NumPy, pandas, SciPy 1.16, matplotlib. Seeds control Python's `random`, NumPy's global RNG and a per-run `Generator`; the same command reproduces the same numbers.

---

## 9. What goes into the 3-page paper (rubric mapping)

| Rubric item | Content to use |
|---|---|
| **Problem definition (3)** | The RQ + hypothesis in §6, stated verbatim in the Introduction; one sentence on why size-adding vs size-removing operators matter for a distance-to-target objective. |
| **Algorithm (5)** | §2: representation, fitness, (μ+λ) scheme, the two operators contrasted mechanistically (table). *Justified parameters*: §5.2–5.3 — rate 0.75 (plateau ≥ 0.5), 1 event (2 is n.s.), crossover 1.0 (monotonic main effect), truncation (selection had the weakest effect in stage 1). |
| **Methodology (5)** | §3 funnel in 4–5 sentences; the held-out-seed rule; budget 6,060 evaluations; random-search baseline; the racing/winner's-curse point as justification for the design. Figures: convergence, tree size. |
| **Results & discussion (10)** | §6 table + `fig_convergence.png` + `fig_tree_size.png`; link back to the hypothesis via the size mechanism (§5.4: ρ = −0.72, targets 7–25 modules); the interaction (§5.3) as the "unexpected finding"; limitations from §7 (ties, floor, n = 10, per-target breakdown). |
| **Conclusion (5)** | Answer the RQ directly (yes, Δ ≈ 0.5, p ≈ 0.007, mechanism = tree-size collapse); future work: size-fair crossover or an explicit "grow" operator to test the substitution hypothesis; vary target set / cost weights. |
| **Structure (2)** | Keep code details (CLI flags, folders) out of the paper; keep the prose at the level of §2 and §6. |

---

## 10. Open items before the deadline (21 Sep 2026, 09:00)

1. Decide how many variants to present in the paper: all five operators (one convergence plot, richest story) or just the two extremes (subtree vs hoist; simplest). Both are "one aspect".
2. Decide: keep the mutation axis (recommended) or switch to crossover (`none` vs `cxr=1` with mutation fixed; effect 0.41 with subtree, 1.34 with replace — the latter needs justifying). Re-running stage 4 for a different axis takes ≈ 2 minutes.
3. Write the paper in the GECCO19 (ACM `acmart`, `sigconf`) template: cover page (course, assignment, team number, names + IDs, date), Introduction, Methods, Results & Discussion, Conclusions, References — ≤ 3 pages excluding cover and bibliography.
4. Clean the code for submission: remove or clearly mark the unused selection/age options; keep `A1_evolution_tree.py`, `A1_final_analysis.py`, `tree_edit_distance.py`, `target_bodies/`; do **not** ship `__data__` sqlite files (`screening/` still holds ≈ 540 MB of them).
5. Package as `groupnumber.zip` containing a folder of the same name with `groupnumber.pdf` and the code.

References to cite: Koza (1992) *Genetic Programming* (subtree crossover/mutation); Poli, Langdon & McPhee (2008) *A Field Guide to Genetic Programming* (hoist/shrink mutation, bloat); Eiben & Smith (2015) *Introduction to Evolutionary Computing* (selection schemes, (μ+λ)); Zhang & Shasha (1989) tree edit distance; Mann & Whitney (1947); Holm (1979); the course's Ariel repository.
