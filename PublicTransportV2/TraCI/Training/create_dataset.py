from multiprocessing import Pool

from tqdm import tqdm

import wandb
import numpy as np
features = ["mean_speed_in_end_PTL","mean_speed_in_PTL","num_total_vehs","num_vehs_in_PTL","MinNumPass","mean_pass_delay_timestamp"]

def handle_run(run):
    df_history = run.history()
    try:
        if "MinPassNum" in df_history.columns:
            df_history["MinNumPass"] = df_history["MinPassNum"]
        df = df_history[features]
    except:
        return None, None, False
    df = df.astype(float).dropna(subset=["MinNumPass","mean_pass_delay_timestamp"])
    values_to_fill = {"mean_speed_in_end_PTL": 25, "mean_speed_in_PTL": 25, "num_total_vehs": 0, "num_vehs_in_PTL": 0}
    df = df.fillna(value=values_to_fill)
    X_train = df[features[:-1]]
    y_train = df["mean_pass_delay_timestamp"]
    X_train = X_train.to_numpy()[:-1,:]
    y_train = y_train.to_numpy()[1:]
    return X_train, y_train, True

if __name__ == "__main__":
    api = wandb.Api()
    username = api.default_entity

    exp_names = ["LeftCompDaily","LeftCompScenarios","ClosedLeftCompScenarios",]
    proj_names = [f"{exp_name}_av{av_rate}" for exp_name in exp_names for av_rate in [0.1,0.2,0.3,0.4,0.6,0.8]]

    runs = []
    for proj_name in proj_names:
        runs.extend(api.runs(f"{username}/{proj_name}"))
    results = []
    for run in tqdm(runs):
        X,y,success = handle_run(run)
        if success:
            results.append((X,y))
    X_train = np.concatenate([X for X,y in results])
    y_train = np.concatenate([y for X,y in results])

    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)