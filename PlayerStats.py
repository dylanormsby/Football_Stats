import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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

    # Shooting
    "Sh",
    "SoT",

    # Passing
    "Ast_stats_passing",
    "xAG_stats_passing",
    "xA",
    "KP",
    "PPA",
    "PrgP",

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
            df[stat] /
            df["90s"].replace(0, None)
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

per90_df = per90_df.replace(
    [float("inf"), float("-inf")],
    0
)

per90_df = per90_df.fillna(0)

per90_df = per90_df.add_suffix("/90")

df = pd.concat([df, per90_df], axis=1)

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


# =========================
# STAT GROUPS
# =========================

stat_groups = {

    # =========================
    # BASIC PLAYER INFORMATION
    # =========================

    "Player Information": [
        "Player",
        "Nation",
        "Pos",
        "Squad",
        "Comp",
        "Age",
      
    ],

    # =========================
    # GOALS & ASSISTS
    # =========================

    "Goal Contribution": [
        "Gls",
        "Ast",
        "G+A",
        "G-PK",
        "PK",
        "PKatt",
        "G+A-PK"
    ],


    # =========================
    # EXPECTED METRICS
    # =========================

    "Expected Performance": [
        "xG",
        "npxG",
        "xAG",
        "npxG+xAG",
        "xG+xAG",
        "G-xG",
        "np:G-xG"
    ],


    # =========================
    # SHOOTING
    # =========================

    "Shooting": [
        "Sh",
        "SoT",
        "SoT%",
        "Sh/90",
        "SoT/90",
        "G/Sh",
        "G/SoT",
        "Dist",
        "FK",
        "PK_stats_shooting",
        "PKatt_stats_shooting",
        "xG_stats_shooting",
        "npxG_stats_shooting",
        "npxG/Sh"
    ],


    # =========================
    # PROGRESSION
    # =========================

    "Ball Progression": [
        "PrgC",
        "PrgP",
        "PrgR",
        "PrgDist",
        "PrgDist_stats_possession",
        "PrgC_stats_possession",
        "PrgP_stats_passing",
        "PrgR_stats_possession"
    ],


    # =========================
    # PASSING
    # =========================

    "Passing": [
        "Cmp",
        "Att",
        "Cmp%",
        "TotDist",
        "Ast_stats_passing",
        "xAG_stats_passing",
        "xA",
        "A-xAG",
        "KP",
        "1/3",
        "PPA",
        "CrsPA",
        "PrgP_stats_passing",
        "Live",
        "Dead",
        "FK_stats_passing_types",
        "TB",
        "Sw",
        "Crs",
        "TI",
        "CK",
        "In",
        "Out",
        "Str",
        "Cmp_stats_passing_types"
    ],


    # =========================
    # POSSESSION / CARRYING
    # =========================

    "Possession & Carrying": [
        "Touches",
        "Def Pen",
        "Def 3rd_stats_possession",
        "Mid 3rd_stats_possession",
        "Att 3rd_stats_possession",
        "Att Pen",
        "Live_stats_possession",
        "Att_stats_possession",
        "Succ",
        "Succ%",
        "Tkld",
        "Tkld%",
        "Carries",
        "TotDist_stats_possession",
        "CPA",
        "Mis",
        "Dis",
        "Rec",
        "1/3_stats_possession"
    ],


    # =========================
    # CHANCE CREATION
    # =========================

    "Chance Creation": [
        "SCA",
        "SCA90",
        "PassLive",
        "PassDead",
        "TO",
        "Sh_stats_gca",
        "GCA",
        "GCA90"
    ],


    # =========================
    # DEFENDING
    # =========================

    "Defensive Actions": [
        "Tkl",
        "TklW",
        "Def 3rd",
        "Mid 3rd",
        "Att 3rd",
        "Att_stats_defense",
        "Tkl%",
        "Lost",
        "Blocks_stats_defense",
        "Sh_stats_defense",
        "Pass",
        "Int",
        "Tkl+Int",
        "Clr",
        "Err"
    ],


    # =========================
    # DUELS / PHYSICAL
    # =========================

    "Duels & Physical": [
        "Won",
        "Lost_stats_misc",
        "Won%",
        "Recov",
        "Fls",
        "Fld_stats_misc",
        "CrdY_stats_misc",
        "CrdR_stats_misc",
        "2CrdY"
    ],


    # =========================
    # DISCIPLINE
    # =========================

    "Discipline": [
        "CrdY",
        "CrdR",
        "2CrdY",
        "Fls",
        "Fld_stats_misc"
    ],


    # =========================
    # GOALKEEPING BASIC
    # =========================

    "Goalkeeping": [
        "GA",
        "GA90",
        "SoTA",
        "Saves",
        "Save%",
        "W",
        "D",
        "L",
        "CS",
        "CS%",
        "PKatt_stats_keeper",
        "PKA",
        "PKsv",
        "PKm",
        "PSxG",
        "PSxG/SoT",
        "PSxG+/-",
        "/90"
    ],


    # =========================
    # ADVANCED GOALKEEPING
    # =========================

    "Advanced Goalkeeping": [
        "Cmp_stats_keeper_adv",
        "Att_stats_keeper_adv",
        "Cmp%_stats_keeper_adv",
        "Att (GK)",
        "Thr",
        "Launch%",
        "AvgLen",
        "Opp",
        "Stp",
        "Stp%",
        "#OPA",
        "#OPA/90",
        "AvgDist"
    ]
}
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
archetype_descriptions = {

    "DF": {

        "Ball Playing Defender":
            "Centre back who progresses play. High passing, carries and progressive actions.",

        "Stopper":
            "Physical defender. High tackles, interceptions, clearances and aerial duels.",

        "Full Back":
            "Wide defender. High progression, carries and attacking involvement."
    },


    "MF": {

        "Playmaker":
            "Creative midfielder. High xA, key passes, progressive passing and chance creation.",

        "Box To Box":
            "Complete midfielder. High progression, carries, defensive work and involvement.",

        "Ball Winner":
            "Defensive midfielder. High tackles, interceptions, recoveries and duels."
    },


    "FW": {

        "Clinical Finisher":
            "Elite goalscorer. High goals, xG and finishing efficiency.",

        "Complete Forward":
            "All-round attacker combining finishing, creation and progression.",

        "Pressing Forward":
            "High work-rate attacker who creates through pressing.",

        "Creator":
            "Attacker focused on assists and chance creation."
    },


    "GK": {

        "Shot Stopper":
            "Goalkeeper focused on saves and shot prevention.",

        "Sweeper Keeper":
            "Aggressive goalkeeper with distribution and actions outside box."
    }
}
archetype_stats = {

    # =========================
    # FORWARDS
    # =========================

    "FW": {

        "Clinical Finisher": [
            "Gls/90",
            "xG/90",
            "npxG/90",
            "SoT/90",
            "G/Sh",
            "G/SoT"
        ],


        "Complete Forward": [
            "G+A/90",
            "xG/90",
            "xAG/90",
            "PrgR/90",
            "Carries/90",
            "SCA90",
            "GCA90"
        ],


        "Pressing Forward": [
            "Sh/90",
            "SCA90",
            "GCA90",
            "Carries/90",
            "Succ%",
            "Recov/90"
        ],


        "Creator": [
            "Ast/90",
            "xAG/90",
            "xA/90",
            "KP/90",
            "PPA/90",
            "SCA90",
            "GCA90"
        ]

    },



    # =========================
    # MIDFIELDERS
    # =========================

    "MF": {

        "Playmaker": [
            "Ast/90",
            "xAG/90",
            "xA/90",
            "KP/90",
            "PPA/90",
            "PrgP/90",
            "SCA90",
            "GCA90"
        ],


        "Box To Box": [
            "G+A/90",
            "PrgC/90",
            "PrgP/90",
            "Carries/90",
            "Tkl/90",
            "Int/90",
            "Recov/90",
            "90s"
        ],


        "Ball Winner": [
            "Tkl/90",
            "TklW/90",
            "Int/90",
            "Tkl+Int/90",
            "Recov/90",
            "Won/90",
            "Won%"
        ]

    },



    # =========================
    # DEFENDERS
    # =========================

   "DF": {

    "Ball Playing Defender": [
        "Cmp%",
        "TotDist/90",
        "PrgP/90",
        "PrgC/90",
        "Carries/90",
        "Succ%"
    ],


    "Stopper": [
        "Tkl/90",
        "TklW/90",
        "Int/90",
        "Tkl+Int/90",
        "Clr/90",
        "Blocks_stats_defense/90",
        "Won/90",
        "Won%",
        "Recov/90"
    ],


    "Full Back": [
        "PrgC/90",
        "PrgP/90",
        "PrgR/90",
        "Carries/90",
        "Succ%",
        "CPA/90",
        "SCA90",
        "Cmp%"
    ]},



    # =========================
    # GOALKEEPERS
    # =========================

    "GK": {

        "Shot Stopper": [
            "GA90",
            "Save%",
            "CS%",
            "PSxG/SoT",
            "PSxG+/-",
            "Stp%"
        ],


        "Sweeper Keeper": [
            "PSxG+/-",
            "Stp%",
            "#OPA/90",
            "AvgDist",
            "Launch%",
            "AvgLen",
            "Cmp%_stats_keeper_adv"
        ]

    }

}

player_profiles = {

    "FW": [
        "Clinical Finisher",
        "Complete Forward",
        "Pressing Forward",
        "Creator"
    ],

    "MF": [
        "Playmaker",
        "Box To Box",
        "Ball Winner"
    ],

    "DF": [
        "Ball Playing Defender",
        "Stopper",
        "Full Back"
    ],

    "GK": [
        "Shot Stopper",
        "Sweeper Keeper"
    ]
}


# =========================
# POSITION SELECTION
# =========================

positions = sorted(df["Main_Position"].unique())


def choose_position():

    print("\nPick a position:")

    for index, position in enumerate(positions, start=1):
        print(f"{index}: {position}")

    choice = int(input("> "))

    selected_position = positions[choice - 1]

    return selected_position

def choose_profile(position):

    profiles = archetype_descriptions[position]

    print("\nChoose player profile:\n")

    for index, (profile, description) in enumerate(profiles.items(), start=1):
        print(f"{index}: {profile}")
        print(f"   {description}\n")

    choice = int(input("> "))

    selected_profile = list(profiles.keys())[choice - 1]

    return selected_profile

# =========================
# MAIN PROGRAM
# =========================

def rank_players():

    # Ask the user what position they want to analyse
    selected_position = choose_position()


    # Ask the user what type of player they want
    selected_profile = choose_profile(selected_position)


    # The number of clusters is based on how many player archetypes exist
    number_of_clusters = len(archetype_stats[selected_position])


    # Filter the main dataframe so we only keep players from the chosen position
    players = df[df["Main_Position"] == selected_position].copy()
    #low minutes ruining clustering
    players = players[players["90s"] >= 5]

    # Get the statistics that are relevant for this position
    #
    # We do NOT want every statistic.
    selected_stats = position_features[selected_position]


    # Remove any statistics that don't exist in our CSV
    selected_stats = [
        stat for stat in selected_stats
        if stat in players.columns
    ]

    print("Available stats:")
    print(selected_stats)
    X = players[selected_stats].copy()

    X = X.select_dtypes(include="number")

    X = X.replace(
        [float("inf"), float("-inf")],
        0
    )

    X = X.fillna(0)

    # StandardScaler puts every statistic onto the same scale.- everything equal 
    scaler = StandardScaler()


    # Convert our stats into scaled values
    X_scaled = scaler.fit_transform(X)


    # Create the KMeans machine learning model- this will cluster the relevant groups based on the stats we care about 
    kmeans = KMeans(
        n_clusters=number_of_clusters,
        random_state=42
    )


    # Run the clustering algorithm
    clusters = kmeans.fit_predict(X_scaled)
    cluster_centers = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=X.columns
    )
    # Evaluate how well separated the clusters are
    silhouette = silhouette_score(
        X_scaled,
        clusters
    )

    # Match the players dataframe back to the players used in clustering
    players = players.loc[X.index].copy()


    # Add the cluster number to every player
    players["Cluster"] = clusters



    # Calculate the average statistics of each cluster.
    cluster_summary = (
        players
        .groupby("Cluster")[X.columns]
        .mean()
    )
    cluster_to_archetype = {}

    remaining_clusters = list(cluster_centers.index)

    for archetype, stats in archetype_stats[selected_position].items():

        stats = [s for s in stats if s in cluster_centers.columns]

        if len(stats) == 0:
            print(f"No matching stats found for {archetype}")
            continue

        best_cluster = None
        best_score = -999

        for cluster in remaining_clusters:

            score = cluster_centers.loc[cluster, stats].mean()

            if score > best_score:
                best_score = score
                best_cluster = cluster

        if best_cluster is not None:
            cluster_to_archetype[best_cluster] = archetype
            remaining_clusters.remove(best_cluster)

    players["Archetype"] = players["Cluster"].map(cluster_to_archetype)

    players = players[
    players["Archetype"] == selected_profile
    ].copy()

    ranking_stats = [
        s for s in archetype_stats[selected_position][selected_profile]
        if s in X.columns
    ]

    cluster_id = [
        k for k, v in cluster_to_archetype.items()
        if v == selected_profile
    ][0]

    centre = cluster_centers.loc[cluster_id, ranking_stats]

    scaled_players = pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index
    )

    scaled_players = scaled_players.loc[players.index]

    players["Score"] = -(
        (
            scaled_players[ranking_stats] - centre
        ) ** 2
    ).sum(axis=1)

    players = players.sort_values(
        "Score",
        ascending=False
    )

    print(
        players[
            ["Player", "Squad", "Score"] + ranking_stats
        ].head(25)
    )

rank_players()