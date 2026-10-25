"""DATA 5600 Project 02 — champion-checklist parameter recovery simulation.

Run: uv run --with numpy --with pandas --with statsmodels python code/project02_champion_simulation.py
SYNTHETIC outcomes: independent Bernoulli draws are a simplified data story;
real NCAA tournaments have exactly one champion per year.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(5600)
n = 4800
quality = rng.normal(size=n)

def logistic(v):
    return 1 / (1 + np.exp(-v))

# All 15 Macfax checks are represented by three thematic pass counts:
# Strength (5): trapezoid, balanced dominance, title favorite,
#   elite efficiency, elite AdjEM
# Resume (3): win percentage, AP week 6, WAB
# Fundamentals (7): 3P%, eFG margin, FTR margin, rebounding edge,
#   turnover edge, four factor index, FT%
strength = rng.binomial(5, logistic(-.65 + .90 * quality))
resume = rng.binomial(3, logistic(-.85 + .90 * quality))
fundamentals = rng.binomial(7, logistic(-.85 + .55 * quality))

x = sm.add_constant(pd.DataFrame({
    "strength_checks": strength,
    "resume_checks": resume,
    "four_factor_shooting_checks": fundamentals,
}), has_constant="add")
true = pd.Series({
    "const": -5.90, "strength_checks": .38,
    "resume_checks": .30, "four_factor_shooting_checks": .18,
})
probability = logistic(x @ true)
champion = rng.binomial(1, probability, size=n)
fit = sm.GLM(champion, x, family=sm.families.Binomial()).fit()
ci = fit.conf_int()
recovery = pd.DataFrame({
    "true": true, "estimated": fit.params,
    "ci_lower": ci[0], "ci_upper": ci[1],
})
recovery["CI_contains_true"] = (
    (recovery.ci_lower <= recovery.true) &
    (recovery.true <= recovery.ci_upper)
)

out = Path("data")
out.mkdir(exist_ok=True)
pd.DataFrame({
    "synthetic_team_id": np.arange(1, n + 1),
    "strength_checks": strength,
    "resume_checks": resume,
    "four_factor_shooting_checks": fundamentals,
    "passed_count": strength + resume + fundamentals,
    "champion": champion,
}).to_csv(out / "project02_simulated_champion_checklist.csv", index=False)
recovery.to_csv(out / "project02_parameter_recovery.csv")

print(f"Simulated teams: {n:,}; champion flags: {champion.sum()} ({champion.mean():.2%})")
print(recovery.round(3).to_string())
print("True values in all 95% CIs:", bool(recovery.CI_contains_true.all()))
