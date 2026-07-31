import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import zscore
# =========================
# LOAD DATA
# =========================

df = pd.read_csv("players_data-2024_2025.csv")
# =========================
# CREATE PER 90 STATS
# =========================

stats_to_convert = [

    # Attack
    "Gls",
    "Ast",
    "G+A",
    "xG",
    "npxG",
    "xAG",
    "G-xG",
    "CPA",

    # Passing
    "Ast_stats_passing",
    "xAG_stats_passing",
    "xA",
    "KP",
    "PPA",
    "PrgP",
    # Passing distance
    "TotDist",
    "PrgDist",

    # Possession distance
    "TotDist_stats_possession",
    "PrgDist_stats_possession",

    # Progression
    "PrgC",
    "PrgR",
    "Carries",

    # Defence
    "Tkl",
    "TklW",
    "Int",
    "Tkl+Int",
    "Clr",
    "Blocks_stats_defense",
    "Sh_stats_defense",

    # Physical
    "Won",
    "Recov"

]


per90_df = pd.DataFrame(index=df.index)

for stat in stats_to_convert:

    if stat in df.columns:

        per90_df[f"{stat}/90"] = (
            pd.to_numeric(df[stat], errors="coerce") /
            pd.to_numeric(df["90s"], errors="coerce").replace(0, None)
)


per90_df = (
    per90_df
    .replace([float("inf"), float("-inf")], 0)
    .fillna(0)
)


df = pd.concat(
    [df, per90_df],
    axis=1
)


per90_df = per90_df.fillna(0)

# =========================
# POSITION CLEANING
# =========================

def clean_position(pos):
    if "GK" in pos:
        return "GK"
    elif "DF" in pos:
        return "DF"
    elif "MF" in pos:
        return "MF"
    elif "FW" in pos:
        return "FW"
    else:
        return "Unknown"


df["Main_Position"] = df["Pos"].fillna("").apply(clean_position)

position_features = {

    "FW": [

        # Finishing
        "Gls/90",
        "G+A/90",
        "xG/90",
        "npxG/90",
        "G-xG/90",

        # Shooting profile
        "Sh/90",
        "SoT/90",
        "SoT%",
        "G/Sh",
        "G/SoT",

        # Creation
        "Ast/90",
        "xAG/90",
        "SCA90",
        "GCA90",

        # Carrying / progression
        "PrgC/90",
        "PrgR/90",
        "Carries/90",
        "CPA/90",

    ],


    "MF": [

        # Creativity
        "Ast/90",
        "xAG/90",
        "xA/90",
        "KP/90",
        "PPA/90",
        "SCA90",
        "GCA90",

        # Progression
        "PrgP/90",
        "PrgC/90",
        "PrgR/90",

        # Passing
        "Cmp%",
        "TotDist/90",

        # Carrying
        "Carries/90",
        "Succ%",
        
        # Defence
        "Tkl/90",
        "Int/90",
        "Tkl+Int/90",
        "Recov/90",

    ],


    "DF": [

        # Defensive activity
        "Tkl/90",
        "TklW/90",
        "Int/90",
        "Tkl+Int/90",
        "Clr/90",
        "Blocks_stats_defense/90",

        # Physical defending
        "Won/90",
        "Won%",
        "Recov/90",

        # Ball playing
        "Cmp%",
        "TotDist/90",
        "PrgP/90",
        "PrgC/90",
        "Carries/90",

    ],


    "GK": [

    "GA90",
    "Save%",
    "CS%",
    "PSxG",
    "PSxG/SoT",
    "PSxG+/-",

    "Stp%",
    "#OPA/90",
    "AvgDist",

    "Launch%",
    "AvgLen",

    "90s"
]
}
archetype_stats = {

    # =========================
    # FORWARDS
    # =========================

    "FW": {

    "Clinical Finisher": {
        "Gls/90": 3,
        "xG/90": 2,
        "npxG/90": 2,
        "SoT/90": 2,
        "G/Sh": 3,
        "G/SoT": 3
    },


    "Complete Forward": {
        "G+A/90": 3,
        "xG/90": 2,
        "xAG/90": 2,
        "PrgR/90": 2,
        "Carries/90": 2,
        "SCA90": 2,
        "GCA90": 3
    },


    "Pressing Forward": {
        "Sh/90": 2,
        "SCA90": 2,
        "GCA90": 1,
        "Carries/90": 2,
        "Succ%": 2,
        "Recov/90": 3,
        "Tkl/90": 3,
        "Int/90": 2
    },


    "Creator": {
        "Ast/90": 3,
        "xAG/90": 3,
        "xA/90": 3,
        "KP/90": 3,
        "PPA/90": 2,
        "SCA90": 2,
        "GCA90": 2
    }

},



    # =========================
    # MIDFIELDERS
    # =========================

    "MF": {

    "Playmaker": {
        "Ast/90": 2,
        "xAG/90": 3,
        "xA/90": 3,
        "KP/90": 3,
        "PPA/90": 2,
        "PrgP/90": 3,
        "SCA90": 1,
        "GCA90": 2
    },


    "Box To Box": {
        "G+A/90": 1,
        "PrgC/90": 3,
        "PrgP/90": 2,
        "Carries/90": 3,
        "Tkl/90": 2,
        "Int/90": 2,
        "Recov/90": 3,
        "Won/90": 2
    },


    "Ball Winner": {
        "Tkl/90": 3,
        "TklW/90": 2,
        "Int/90": 2,
        "Tkl+Int/90": 3,
        "Recov/90": 3,
        "Won/90": 3,
        "Won%": 2
    }

},

    



    # =========================
    # DEFENDERS
    # =========================

   "DF": {

    "Ball Playing Defender": {
        "Cmp%":3,
        "TotDist/90":2,
        "PrgP/90":3,
        "PrgC/90":1,
        "Carries/90":2,
        
    },


    "Stopper": {
        "Tkl/90":1,
        "TklW/90":3,
        "Int/90":2,
        "Tkl+Int/90":2,
        "Clr/90":3,
        "Blocks_stats_defense/90":3,
        "Won/90":3,
        "Won%":3,
        "Recov/90":3
   },


    "Wing Back": {
        "PrgC/90":3,
        "PrgP/90":2,
        "PrgR/90":1,
        "Carries/90":3,
        "Succ%":1,
        "CPA/90":2,
        "SCA90":2,
        "Cmp%":1
},

  "Progressive Defender": {
        "PrgC/90": 3,
        "Carries/90": 3,
        "PrgR/90": 2,
        "Succ%": 2,
        "TotDist/90": 2
    }
},



    # =========================
    # GOALKEEPERS
    # =========================

 "GK": {

    "Shot Stopper": {
        "Save%": 3,
        "PSxG+/-": 3,
        "CS%": 2,
        "PSxG/SoT": 2,
        "Stp%": 2
    },


    "Sweeper Keeper": {
        "#OPA/90": 5,
        "AvgDist": 4,
        "AvgLen": 3,
        "Launch%": 2,
        "Cmp%_stats_keeper_adv": 3,
        "PSxG+/-": 1
    },


    "Traditional Keeper": {
        "GA90": 2,
        "Save%": 2,
        "CS%": 2,
        "PSxG": 4,
        "90s": 1
    }

}

}


def create_cluster_zscores(cluster_summary):

    # convert everything to numeric
    numeric_summary = cluster_summary.apply(
        lambda x: pd.to_numeric(x, errors="coerce")
    )

    # remove columns that became completely NaN
    numeric_summary = numeric_summary.dropna(
        axis=1,
        how="all"
    )

    zscores = numeric_summary.apply(
        zscore
    )

    return zscores

# =========================
# AUTOMATIC PLAYER CLUSTERING
# =========================
def name_clusters(cluster_summary, position):

    archetypes = archetype_stats[position]

    # lower is better stats
    reverse_stats = {
        "GA90",
    }

    archetype_scores = {}

    for cluster in cluster_summary.index:

        cluster_values = cluster_summary.loc[cluster]

        scores = {}

        for archetype, stats in archetypes.items():

            score = 0

            for stat, weight in stats.items():

                if stat in cluster_summary.columns:

                    value = cluster_values[stat]

                    if stat in reverse_stats:
                        value *= -1

                    score += value * weight


            scores[archetype] = score

        print(cluster, scores)
        best_match = max(
            scores,
            key=scores.get
        )

        archetype_scores[cluster] = best_match


    return archetype_scores


def create_player_clusters():

    for position in ["DF", "MF", "FW", "GK"]:

        print(f"Processing {position}")

        players = df[
            df["Main_Position"] == position
        ].copy()


        players = players[
            players["90s"] >= 5
        ]


        # GET POSITION SPECIFIC FEATURES
        features = position_features[position]
        # GET POSITION SPECIFIC FEATURES


        numeric = players[features]


        numeric = numeric.fillna(0)


        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(
            numeric
        )


        if position == "FW":
            clusters = 5

        elif position == "MF":
            clusters = 5

        elif position == "DF":
            clusters = 4

        elif position == "GK":
            clusters = 3

        missing = []

        for pos, feature_list in position_features.items():
            for f in feature_list:
                if f not in df.columns:
                    missing.append((pos,f))

        kmeans = KMeans(
            n_clusters=clusters,
            random_state=42,
            n_init=20
        )


        labels = kmeans.fit_predict(
            X_scaled
        )


        players["Cluster"] = labels

        cluster_summary = (
            players
            .groupby("Cluster")[features]
            .mean()
            .round(2)
        )

        cluster_zscores = create_cluster_zscores(
        cluster_summary
        )       

        print("\n", position, "cluster profiles")
        print(cluster_summary)

        print("\nCluster sizes")
        print(players["Cluster"].value_counts())

        cluster_names = name_clusters(
            cluster_zscores,
            position
        )

        print("\n", position, "cluster names")
        for cluster, name in cluster_names.items():
            print(f"Cluster {cluster} -> {name}")

        df.loc[
            players.index,
            "Cluster"
        ] = labels
        df.loc[
            players.index,
            "Archetype"
        ] = [
            cluster_names[c]
            for c in labels
        ]

        print("\n", position, "cluster profiles")
        print(cluster_summary)

        print(
            position,
            "clusters created:",
            clusters
        )


create_player_clusters()


df.to_csv(
    "players_master.csv",
    index=False
)


print("Finished!")