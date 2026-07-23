import pandas as pd

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("players_data-2024_2025.csv")


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

    "Attacking Output": [
        "Gls",
        "Ast",
        "G+A",
        "G-PK",
        "PK",
        "PKatt",
        "G+A-PK"
    ],

    "Expected Attacking": [
        "xG",
        "npxG",
        "xAG",
        "npxG+xAG",
        "xG+xAG",
        "G-xG"
    ],

    "Shooting": [
        "Sh",
        "SoT",
        "SoT%",
        "Sh/90",
        "SoT/90",
        "G/Sh",
        "G/SoT"
    ],

    "Progression": [
        "PrgC",
        "PrgP",
        "PrgR",
        "PrgDist"
    ],

    "Passing": [
        "Cmp",
        "Att",
        "Cmp%",
        "TotDist",
        "xA",
        "KP",
        "PPA"
    ],

    "Chance Creation": [
        "SCA",
        "SCA90",
        "GCA",
        "GCA90"
    ],

    "Defending": [
        "Tkl",
        "TklW",
        "Tkl%",
        "Int",
        "Tkl+Int",
        "Clr",
        "Blocks_stats_defense"
    ],

    "Possession": [
        "Touches",
        "Carries",
        "Succ",
        "Succ%",
        "Rec",
        "Mis",
        "Dis"
    ],

    "Goalkeeping": [
        "GA",
        "GA90",
        "Saves",
        "Save%",
        "CS",
        "PSxG"
    ]
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

    profiles = player_profiles[position]

    print("\nChoose player profile:")

    for index, profile in enumerate(profiles, start=1):
        print(f"{index}: {profile}")

    choice = int(input("> "))

    selected_profile = profiles[choice - 1]

    return selected_profile

# =========================
# MAIN PROGRAM
# =========================

def rank_players():

    selected_position = choose_position()
    selected_profile = choose_profile(selected_position)
    players = df[df["Main_Position"] == selected_position]

    print("\nYou picked:", selected_position)
    print(selected_profile)



rank_players()