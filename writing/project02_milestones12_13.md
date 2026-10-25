# Project 02 — NCAA National Champion Checklist (Milestones 12 & 13)

## Scope and decision
Keep the existing Macfax 15-check NCAA Champion Checklist and its historical thresholds. **Only predict the national champion**, not Final Four advancement. One observation in the eventual real analysis = one NCAA tournament team's pre-tournament season. Binary dependent variable: champion = 1 if the team won the national title, else 0.

**Business question:** Which teams should Macfax classify and feature as plausible national championship contenders based on the checklist? Goal: avoid omitting the eventual champion without labeling almost everybody a contender.

The active source: https://github.com/spencerjmac/macfax/blob/main/backend/api/crystal_ball_views.py . It ranks teams by passedCount / 15 (a score, not a probability). The proposed logistic model keeps all 15 conditions, maps them into three interpretable score groups, and estimates the probability that a team wins the title.

## Milestone 12: Piecewise decision loss
Let action a=1 mean 'feature as championship contender'; actual outcome y=1 means 'national champion'. Loss is illustrative editorial cost units, **not measured dollars**:

- a=1, y=1: 0 (correctly featured champion)
- a=1, y=0: 1 (featured a team that did not win)
- a=0, y=1: 20 (missed the eventual champion)
- a=0, y=0: 0 (correctly did not feature)

Expected loss if featured: (1-p)*1.
Expected loss if not featured: p*20.
Feature if p > 1/21 (approximately 4.76%). Sensitivity-analysis will try other ratios.

## Milestone 13: Data story and working simulation
Macfax ingests basketball game stats, calculates advanced ratings and the four-factor statistics, uses pre-tournament snapshots to evaluate all D1 teams on 15 checks, and has championship results and historical checklist backtest exports. Proposed dataset: **only NCAA tournament teams**, across available 2005–2026 tournaments except the canceled 2020 event. Confirm completeness and no post-tournament predictor leakage before fitting.

- Strength (5 checks): trapezoid, balanced dominance, title favorite, elite efficiency, elite AdjEM.
- Resume (3): win percentage, Week 6 AP ranking, WAB.
- Fundamentals (7): 3P%, eFG margin, FTR margin, rebounding edge, turnover edge, Four Factor Index, FT%.

Model concept: logit(P(champion)) = beta0 + beta1*(strength count) + beta2*(resume count) + beta3*(fundamentals count). Thus all 15 checklist rules are retained but the three groups have learned weights.

We simulated 4,800 independent hypothetical teams (not real data) and binary champion flags with known parameters. The Python model recovered parameters within 95% intervals in the tested run.

| Term | True beta | Estimated beta |
|---|---:|---:|
| Intercept | -5.900 | -5.974 |
| Strength count | 0.380 | 0.396 |
| Resume count | 0.300 | 0.384 |
| Fundamentals count | 0.180 | 0.129 |

Simulation event rate: 67 / 4,800 = 1.40%. This is a Bernoulli recovery demonstration. Real tournaments have exactly one winner each, and cross-season and within-season dependence plus few historical champions require discussion.

Code: `code/project02_champion_simulation.py`.

Run from project root:
```bash
uv run --with numpy --with pandas --with statsmodels python code/project02_champion_simulation.py
```
Produces `data/project02_simulated_champion_checklist.csv` and `data/project02_parameter_recovery.csv`.

## Important limitations / next steps
- Only about 21 title winners in the historical seasons: avoid fitting 15 separate unpenalized coefficients; keep model parsimonious.
- Existing checklist thresholds were historically selected using champions. Train/test or walk-forward evaluation must avoid using later title outcomes in threshold selection.
- The original checklist score is not a probability of winning. Logistic predictions need calibration checks; they do not automatically sum to 100% within a season.
- Validate tournament seeds, complete champion labels, historical snapshots, independence assumptions, and whether the original Project 01 data overlap.
- Compare the raw 15-point checklist with the logistic weighting when real data become available.
