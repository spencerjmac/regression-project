"""DATA 5600 Project 02: NBA rookie 5-year longevity simulation."""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(5600)
n = 3500
games = rng.integers(20, 83, n)
minutes = np.clip(rng.normal(19, 7, n), 4, 38)
points = np.clip(0.5 + .42 * minutes + rng.normal(0, 3, n), 0, None)
rebounds = np.clip(.3 + .15 * minutes + rng.normal(0, 1.4, n), 0, None)
assists = np.clip(.2 + .10 * minutes + rng.normal(0, 1, n), 0, None)
x = pd.DataFrame({
    "games_z": (games-50)/20, "minutes_z": (minutes-19)/8,
    "points_z": (points-8.5)/5, "rebounds_z": (rebounds-3.2)/2,
    "assists_z": (assists-2.1)/1.5
})
truth = pd.Series({"const":-.20, "games_z":.65, "minutes_z":.50,
                   "points_z":.35, "rebounds_z":.30, "assists_z":.15})
X = sm.add_constant(x, has_constant="add")
p = 1/(1+np.exp(-(X @ truth)))
y = rng.binomial(1,p,n)
data = pd.DataFrame({"player_id": np.arange(1,n+1), "GP":games, "MIN":minutes,
                     "PTS":points, "REB":rebounds, "AST":assists, "TARGET_5Yrs":y})
path=Path("data/project02_simulated_rookies.csv")
path.parent.mkdir(exist_ok=True)
data.to_csv(path,index=False)
fit=sm.GLM(y,X,family=sm.families.Binomial()).fit()
recovery=pd.DataFrame({"true":truth,"estimated":fit.params,
                       "ci_lower":fit.conf_int()[0],"ci_upper":fit.conf_int()[1]})
print("Players:",n,"five-year rate:",round(y.mean(),3))
print(recovery.round(3).to_string())
print("All CIs contain truth:",bool(((recovery.ci_lower<=recovery.true)&(recovery.true<=recovery.ci_upper)).all()))
print("Saved:",path)
