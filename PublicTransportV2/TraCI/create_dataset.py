from multiprocessing import Pool

from tqdm import tqdm

import wandb
import numpy as np
features = ["mean_speed_in_end_PTL","mean_speed_in_PTL","num_total_vehs","num_vehs_in_PTL","MinNumPass"]

def handle_run(run):
    df_history = run.history()
    if "MinPassNum" in df_history.columns:
        df_history["MinNumPass"] = df_history["MinPassNum"]
    X_train = df_history[features]
    X_train.dropna(subset=["MinNumPass"], inplace=True)
    values_to_fill = {"mean_speed_in_end_PTL": 25, "mean_speed_in_PTL": 25, "num_total_vehs": 0, "num_vehs_in_PTL": 0}
    X_train.fillna(value=values_to_fill, inplace=True)
    y_train = df_history["mean_pass_delay_timestamp"]
    X_train = X_train.to_numpy()[:-1]
    y_train = y_train.to_numpy()[1:]
    return X_train, y_train

if __name__ == "__main__":
    api = wandb.Api()
    username = api.default_entity

    exp_names = ["LeftCompDaily","LeftCompScenarios","ClosedLeftCompScenarios",]
    proj_names = [f"{exp_name}_av{av_rate}" for exp_name in exp_names for av_rate in [0.1,0.2,0.3,0.4,0.6,0.8]]

    runs = []
    for proj_name in proj_names:
        runs.extend(api.runs(f"{username}/{proj_name}"))
    with Pool(10) as pool:
        results = list(tqdm(pool.imap(handle_run, runs), total=len(runs)))
    X_train, y_train = zip(*results)
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)