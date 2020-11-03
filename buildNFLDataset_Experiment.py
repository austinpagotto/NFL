#%%
import pandas as pd

#%%
dfs = ["df1.csv", "df2.csv", "df3.csv", "df4.csv"]
finaldf = pd.DataFrame()

for df in dfs:
    tmp = pd.read_csv(df)
    finaldf = pd.concat([finaldf, tmp])

nflDataset = finaldf.iloc[:, 1:]

#%%
# los angeles = la rams
# st louis= la rams
# san diego = la chargers
# Oakland = Las Vegas
def replaceOldTeams(df, oldTeam, newTeam):
    df.loc[df["homeTeam"] == oldTeam, "homeTeam"] = newTeam
    df.loc[df["awayTeam"] == oldTeam, "awayTeam"] = newTeam
    return df


nflDataset = replaceOldTeams(nflDataset, "Los Angeles", "LA Rams")
nflDataset = replaceOldTeams(nflDataset, "St. Louis", "LA Rams")
nflDataset = replaceOldTeams(nflDataset, "San Diego", "LA Chargers")
nflDataset = replaceOldTeams(nflDataset, "Oakland", "Las Vegas")

teamsList = nflDataset.awayTeam.unique()

statsDf = pd.DataFrame()

# Stats for Team
for team in teamsList:
    tmpHome = nflDataset.loc[(nflDataset["homeTeam"] == team)]
    tmpHome["Team"] = team
    tmpHome["points"] = tmpHome["homeScore"]
    tmpHome["pointsAgainst"] = tmpHome["awayScore"]
    tmpHome["passingYards"] = tmpHome["homePassing"]
    tmpHome["rushingYards"] = tmpHome["homeRushing"]
    tmpHome["passingYardsAgainst"] = tmpHome["awayPassing"]
    tmpHome["rushingYardsAgainst"] = tmpHome["awayRushing"]
    tmpHome["isHome"] = 1
    tmpHome["isWinner"] = tmpHome["homeScore"] > tmpHome["awayScore"]
    tmpAway = nflDataset.loc[(nflDataset["awayTeam"] == team)]
    tmpAway["Team"] = team
    tmpAway["points"] = tmpAway["awayScore"]
    tmpAway["pointsAgainst"] = tmpAway["homeScore"]
    tmpAway["passingYards"] = tmpAway["awayPassing"]
    tmpAway["rushingYards"] = tmpAway["awayRushing"]
    tmpAway["passingYardsAgainst"] = tmpAway["homePassing"]
    tmpAway["rushingYardsAgainst"] = tmpAway["homeRushing"]
    tmpAway["isWinner"] = tmpAway["homeScore"] < tmpAway["awayScore"]
    tmp = pd.concat([tmpHome, tmpAway])
    tmp = tmp.sort_values(by="GameID")
    tmp["isHome"] = tmp["isHome"].fillna(0)
    tmp["isWinner"].fillna(0)
    tmp["isWinner"] = tmp["isWinner"].astype(int)
    tmp["rolling10GamePoints"] = tmp["points"].rolling(10).mean().shift()
    tmp["rolling10GamePointsAgainst"] = tmp["pointsAgainst"].rolling(10).mean().shift()
    tmp["rolling10Record"] = tmp["isWinner"].rolling(10).mean().shift()
    tmp["rolling5Record"] = tmp["isWinner"].rolling(5).mean().shift()
    tmp["rolling10Rushing"] = tmp["rushingYards"].rolling(10).mean().shift()
    tmp["rolling10Passing"] = tmp["passingYards"].rolling(10).mean().shift()
    tmp["rolling10RushingAgainst"] = (
        tmp["rushingYardsAgainst"].rolling(10).mean().shift()
    )
    tmp["rolling10PassingAgainst"] = (
        tmp["passingYardsAgainst"].rolling(10).mean().shift()
    )
    tmp["date2"] = pd.to_datetime(tmp["date"])
    statsDf = pd.concat([statsDf, tmp])


# %%

opponentIsHome = statsDf.loc[statsDf["homeTeam"] != statsDf["Team"]]
opponentIsAway = statsDf.loc[statsDf["awayTeam"] != statsDf["Team"]]

for i in statsDf:
    statsDf = statsDf.rename(columns={i: i + "_opp"})

opponentIsHome = opponentIsHome.merge(
    statsDf.loc[
        :,
        [
            "rolling10GamePoints_opp",
            "rolling10GamePointsAgainst_opp",
            "rolling10Record_opp",
            "rolling5Record_opp",
            "rolling10Rushing_opp",
            "rolling10Passing_opp",
            "rolling10RushingAgainst_opp",
            "rolling10PassingAgainst_opp",
            "date2_opp",
            "Team_opp",
        ],
    ],
    left_on=["date2", "homeTeam"],
    right_on=["date2_opp", "Team_opp"],
)

opponentIsAway = opponentIsAway.merge(
    statsDf.loc[
        :,
        [
            "rolling10GamePoints_opp",
            "rolling10GamePointsAgainst_opp",
            "rolling10Record_opp",
            "rolling5Record_opp",
            "rolling10Rushing_opp",
            "rolling10Passing_opp",
            "rolling10RushingAgainst_opp",
            "rolling10PassingAgainst_opp",
            "date2_opp",
            "Team_opp",
        ],
    ],
    left_on=["date2", "awayTeam"],
    right_on=["date2_opp", "Team_opp"],
)

finalStats = pd.concat([opponentIsHome, opponentIsAway]).sort_values(by="GameID")


# %%
finalStats.to_csv("experimentationDataset.csv")

# %%
