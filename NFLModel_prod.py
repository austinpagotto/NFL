#%%
import pandas as pd
from sklearn.linear_model import LinearRegression

# %%
data = pd.read_csv("experimentationDataset.csv")
data = data.dropna()

independentVars = [
    "isHome",
    "rolling10GamePoints",
    "rolling10GamePointsAgainst",
    "rolling10Record",
    # "rolling5Record",
    "rolling10Rushing",
    "rolling10Passing",
    "rolling10RushingAgainst",
    "rolling10PassingAgainst",
    "rolling10GamePoints_opp",
    "rolling10GamePointsAgainst_opp",
    "rolling10Record_opp",
    # "rolling5Record_opp",
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

X_test["within3Points"] = abs(X_test["points"] - X_test["preds"]) < 3
X_test["within3Points"] = X_test["within3Points"].astype(int)

print(X_test["within3Points"].mean())

# %%
data = pd.read_csv("prodDataset.csv")
teamCols = [
    "Team",
    "isHome",
    "rolling10GamePoints",
    "rolling10GamePointsAgainst",
    "rolling10Record",
    "rolling5Record",
    "rolling10Rushing",
    "rolling10Passing",
    "rolling10RushingAgainst",
    "rolling10PassingAgainst",
]


def predictOnTeams(teamToPredictScore, opponentTeam, isHome):
    teamToPredict = data.loc[data["Team"] == teamToPredictScore].tail(1)
    teamToPredict = teamToPredict.loc[:, teamCols]

    opponentTeam = data.loc[data["Team"] == opponentTeam].tail(1)
    opponentTeam = opponentTeam.loc[:, teamCols]

    for i in opponentTeam:
        opponentTeam = opponentTeam.rename(columns={i: i + "_opp"})

    stackedTeams = pd.concat([teamToPredict, opponentTeam], axis=1)
    stackedTeams["isHome"] = isHome
    stackedTeams["Team"] = teamToPredictScore
    stackedTeams = stackedTeams.drop(columns=["isHome_opp", "Team_opp"])

    stackedTeams = stackedTeams.fillna(stackedTeams.mean()).drop_duplicates()
    stackedTeams = stackedTeams[independentVars]
    print("".join([teamToPredictScore, str(lr.predict(stackedTeams))]))

    stackedTeams["prediction"] = lr.predict(stackedTeams)

    return stackedTeams


# Week 8
Carolina = predictOnTeams("Carolina", "Atlanta", 1)
Atlanta = predictOnTeams("Atlanta", "Carolina", 0)
total = Carolina.prediction + Atlanta.prediction
print(total.values[0])
print("--------------------------")
Pittsburgh = predictOnTeams("Pittsburgh", "Baltimore", 0)
Baltimore = predictOnTeams("Baltimore", "Pittsburgh", 1)
total = Pittsburgh.prediction + Baltimore.prediction
print(total.values[0])
print("--------------------------")
LaRams = predictOnTeams("LA Rams", "Miami", 0)
Miami = predictOnTeams("Miami", "LA Rams", 1)
total = LaRams.prediction + Miami.prediction
print(total.values[0])
print("--------------------------")
Kansas = predictOnTeams("Kansas City", "NY Jets", 1)
Nyj = predictOnTeams("NY Jets", "Kansas City", 0)
total = Kansas.prediction + Nyj.prediction
print(total.values[0])
print("--------------------------")
Minnesota = predictOnTeams("Minnesota", "Green Bay", 0)
GreenBay = predictOnTeams("Green Bay", "Minnesota", 1)
total = Minnesota.prediction + GreenBay.prediction
print(total.values[0])
print("--------------------------")
Indianapolis = predictOnTeams("Indianapolis", "Detroit", 0)
Detroit = predictOnTeams("Detroit", "Indianapolis", 1)
total = Detroit.prediction + Indianapolis.prediction
print(total.values[0])
print("--------------------------")
Cincinnati = predictOnTeams("Cincinnati", "Tennessee", 1)
Tennessee = predictOnTeams("Tennessee", "Cincinnati", 1)
total = Cincinnati.prediction + Tennessee.prediction
print(total.values[0])
print("--------------------------")
NewEngland = predictOnTeams("New England", "Buffalo", 0)
Buffalo = predictOnTeams("Buffalo", "New England", 1)
total = NewEngland.prediction + Buffalo.prediction
print(total.values[0])
print("--------------------------")
LasVegas = predictOnTeams("Las Vegas", "Cleveland", 0)
Cleveland = predictOnTeams("Cleveland", "Las Vegas", 1)
total = LasVegas.prediction + Cleveland.prediction
print(total.values[0])
print("--------------------------")
LaChargers = predictOnTeams("LA Chargers", "Denver", 0)
Denver = predictOnTeams("Denver", "LA Chargers", 1)
total = LaChargers.prediction + Denver.prediction
print(total.values[0])
print("--------------------------")
NewOrleans = predictOnTeams("New Orleans", "Chicago", 0)
Chicago = predictOnTeams("Chicago", "New Orleans", 1)
total = NewOrleans.prediction + Chicago.prediction
print(total.values[0])
print("--------------------------")
SanFrancisco = predictOnTeams("San Francisco", "Seattle", 0)
Seattle = predictOnTeams("Seattle", "San Francisco", 1)
total = Seattle.prediction + SanFrancisco.prediction
print(total.values[0])
print("--------------------------")
Dallas = predictOnTeams("Dallas", "Philadelphia", 0)
Philly = predictOnTeams("Philadelphia", "Dallas", 1)
total = Dallas.prediction + Philly.prediction
print(total.values[0])
print("--------------------------")
TampaBay = predictOnTeams("Tampa Bay", "NY Giants", 0)
Nyg = predictOnTeams("NY Giants", "Tampa Bay", 1)
total = Nyg.prediction + TampaBay.prediction
print(total.values[0])
print("--------------------------")

# %%


# %%
