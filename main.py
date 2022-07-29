import sys
import os
import pickle
import json

import numpy as np

from sklearn.ensemble import RandomForestRegressor

OUTCOMES = ["AWAY", "DRAW", "HOME", "SKIP"]
OPTIMAL_MIN_BET = 1.0
OPTIMAL_MAX_BET = 2.5
N_MATCHES = 15
DATA_DIR = "data"
MODEL_DIR = "models"


def validate_bet(value: float, bets: list) -> str:
    if -0.15 < value < 0.39:
        prediction = 1
    elif value > 0.75:
        prediction = 2
    elif value < -0.85:
        prediction = 0
    else:
        prediction = 3
    if ((prediction != 3) and
        (OPTIMAL_MIN_BET < bets[prediction] < OPTIMAL_MAX_BET)):
        prediction = 3
    return OUTCOMES[prediction]


def form_data(inp_data: list) -> np.array:
    # inp_data = list(map(float, inp_data))
    idx1 = str(inp_data[2]) if stats.get((inp_data[2])) else "-1.0"
    idx2 = str(inp_data[3]) if stats.get((inp_data[3])) else "-1.0"
    idx3 = str(inp_data[4]) if ref_stats.get(inp_data[4]) else "-1.0"

    home_data = np.mean(np.array(stats[idx1][-N_MATCHES:], dtype=int), axis=0)

    away_data = np.mean(np.array(stats[idx2][-N_MATCHES:], dtype=int), axis=0)

    ref_data = (
        list(map(float, ref_stats[idx3]))
    )
    new_data = np.hstack((
        inp_data[0],
        inp_data[5:],
        home_data,
        away_data,
    ))
    return new_data


def refresh_stats(refr_data: list, idx1: str, idx2: str) -> None:
    if stats.get(idx1):
        stats[idx1].append(refr_data[:11:2])
    else:
        stats[idx1] = [refr_data[:11:2]]

    if stats.get(idx2):
        stats[idx2].append(refr_data[1:12:2])
    else:
        stats[idx2] = [refr_data[1:12:2]]


if __name__ == "__main__":
    
    model_fp = os.path.join(MODEL_DIR, "model.pkl")
    with open(model_fp, "rb") as handler:
        model= pickle.load(handler)
    
    stats_fp = os.path.join(DATA_DIR, "stats.json")
    with open(stats_fp, "r") as handler:
        stats = json.load(handler)

    ref_stats_fp = os.path.join(DATA_DIR, "ref_stats.json")
    with open(ref_stats_fp, "r") as handler:
        ref_stats = json.load(handler)

    n_matches = int(sys.stdin.readline())

    for _ in range(n_matches):
        input_data = sys.stdin.readline().split()
        data = form_data(list(map(float, input_data)))

        # Make a prediction
        prediction_values = model.predict(data.reshape(1, -1))

        # Make a bet
        bet = validate_bet(prediction_values, data[1:4])

        sys.stdout.write(f"{bet}\n")
        sys.stdout.flush()

        # Refresh stats
        refresh_data = sys.stdin.readline().split()
        refresh_stats(list(map(int, refresh_data)), str(input_data[2]), str(input_data[3]))
        
