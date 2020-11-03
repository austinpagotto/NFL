#%%
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# %%
data = pd.read_csv("experimentationDataset.csv")
data = data.dropna()

independentVars = [
    "isHome",
    "rolling10GamePoints",
    "rolling10GamePointsAgainst",
    "rolling10Record",
    "rolling10Rushing",
    "rolling10Passing",
    "rolling10RushingAgainst",
    "rolling10PassingAgainst",
    "rolling10GamePoints_opp",
    "rolling10GamePointsAgainst_opp",
    "rolling10Record_opp",
    "rolling10Rushing_opp",
    "rolling10Passing_opp",
    "rolling10RushingAgainst_opp",
    "rolling10PassingAgainst_opp",
]

dependentVar = "points"

# %%
# Split the data into training/testing sets
train = data[
    (pd.to_datetime(data.date2) < pd.to_datetime("2020-09-01"))
    & (pd.to_datetime(data.date2) >= pd.to_datetime("2018-09-01"))
]

test = data[pd.to_datetime(data.date2) >= pd.to_datetime("2020-09-01")]

# Split the targets into training/testing sets
lr = LinearRegression()

X_train = train.loc[:, independentVars]

Y_train = train.loc[:, dependentVar]

lr.fit(X_train, Y_train)

X_test = test.loc[:, independentVars]


# The labels of the model
Y_test = test.loc[:, dependentVar]

preds = lr.predict(X_test)

mae = sum(abs(preds - test[dependentVar])) / test.shape[0]

print(mae)

X_test["preds"] = preds
X_test["points"] = Y_test
X_test["GameId"] = test["GameID"]
X_test["isWinner"] = test["isWinner"]

X_test["within3Points"] = abs(X_test["points"] - X_test["preds"]) < 3
X_test["within3Points"] = X_test["within3Points"].astype(int)

gameLevel = X_test.loc[:, ["GameId", "preds", "points"]]

gameLevel = gameLevel.groupby(["GameId"])[["preds", "points"]].sum()
gameLevel = gameLevel.reset_index()


gameLevel.loc[gameLevel["preds"] <= 20, "predBand"] = "0-20"
gameLevel.loc[gameLevel["preds"] > 20, "predBand"] = "21-30"
gameLevel.loc[gameLevel["preds"] > 30, "predBand"] = "31-40"
gameLevel.loc[gameLevel["preds"] > 40, "predBand"] = "41-50"
gameLevel.loc[gameLevel["preds"] > 50, "predBand"] = "51-60"
gameLevel.loc[gameLevel["preds"] > 60, "predBand"] = "61-70"
gameLevel.loc[gameLevel["preds"] > 70, "predBand"] = "70+"


gameLevel.loc[gameLevel["points"] <= 20, "pointBand"] = "0-20"
gameLevel.loc[gameLevel["points"] > 20, "pointBand"] = "21-30"
gameLevel.loc[gameLevel["points"] > 30, "pointBand"] = "31-40"
gameLevel.loc[gameLevel["points"] > 40, "pointBand"] = "41-50"
gameLevel.loc[gameLevel["points"] > 50, "pointBand"] = "51-60"
gameLevel.loc[gameLevel["points"] > 60, "pointBand"] = "61-70"
gameLevel.loc[gameLevel["points"] > 70, "pointBand"] = "70+"


# gameLevel.loc[gameLevel["preds"] <= 30, "predBand"] = "0-30"
# gameLevel.loc[gameLevel["preds"] > 30, "predBand"] = "31-50"
# gameLevel.loc[gameLevel["preds"] > 50, "predBand"] = "51-70"
# gameLevel.loc[gameLevel["preds"] > 70, "predBand"] = "70+"


# gameLevel.loc[gameLevel["points"] <= 30, "pointBand"] = "0-30"
# gameLevel.loc[gameLevel["points"] > 30, "pointBand"] = "31-50"
# gameLevel.loc[gameLevel["points"] > 50, "pointBand"] = "51-70"
# gameLevel.loc[gameLevel["points"] > 70, "pointBand"] = "70+"

gameLevel["Correct"] = gameLevel["pointBand"] == gameLevel["predBand"]
gameLevel["Correct"] = gameLevel["Correct"].astype(int)

print(gameLevel["Correct"].mean())


# pd.concat([X_test, test], axis=1).to_csv("preds.csv")
# %%
# %%


# %%
