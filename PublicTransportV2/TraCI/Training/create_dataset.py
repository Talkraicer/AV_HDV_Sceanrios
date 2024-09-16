from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

import wandb
import numpy as np

features = ["mean_speed_in_end_PTL", "mean_speed_in_PTL", "num_total_vehs", "num_vehs_in_PTL", "MinNumPass",
            "mean_pass_delay_timestamp"]


def handle_run(run,project_name):
    df_history = run.history()
    try:
        if "MinPassNum" in df_history.columns:
            df_history["MinNumPass"] = df_history["MinPassNum"]
        df = df_history[features]
    except:
        return None, False
    df = df.astype(float).dropna(subset=["MinNumPass", "mean_pass_delay_timestamp"])
    values_to_fill = {"mean_speed_in_end_PTL": 25, "mean_speed_in_PTL": 25, "num_total_vehs": 0, "num_vehs_in_PTL": 0}
    df = df.fillna(value=values_to_fill)
    df["project_name"] = project_name
    return df, True


if __name__ == "__main__":
    api = wandb.Api()
    username = api.default_entity

    exp_names = ["LeftCompDaily", "LeftCompScenarios", "ClosedLeftCompScenarios"]
    proj_names = [f"{exp_name}_av{av_rate}" for exp_name in exp_names for av_rate in [0.1, 0.2, 0.3, 0.4, 0.6, 0.8]]

    dataset = pd.DataFrame(columns=features + ["project_name"])
    for proj_name in proj_names:
        runs = api.runs(f"{username}/{proj_name}")
        for run in tqdm(runs):
            df, success = handle_run(run,proj_name)
            if success:
                dataset = pd.concat([dataset, df])
    dataset.to_pickle("dataset.pkl")
    dataset.to_csv("dataset.csv")


