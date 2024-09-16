import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import pickle
features = ["mean_speed_in_end_PTL","mean_speed_in_PTL","num_total_vehs","num_vehs_in_PTL","MinNumPass"]
target = "mean_pass_delay_timestamp"


def main():
    X = np.load("X_train.npy", allow_pickle=True).astype(float)
    Y = np.load("y_train.npy", allow_pickle=True).astype(float)

    # split to different datasets according to MinNumPass
    values = np.unique(X[:,-1])
    for value in values:
        print(f"MinNumPass = {value}")
        mask = X[:,-1] == value

        X_dataset = X[mask]
        X_dataset = X_dataset[:,:-1]
        Y_dataset = Y[mask]
        X_train, X_test, y_train, y_test = train_test_split(X_dataset, Y_dataset, test_size=0.2)

        os.makedirs("Trees", exist_ok=True)
        model = DecisionTreeRegressor(max_depth=4)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        fig = plt.figure(figsize=(25,20))
        plot_tree(model, feature_names=features[:-1], filled=True)
        plt.savefig(f"Trees/tree_{value}.png")
        plt.show()
        print(f"MSE DecisionTree: {mse}")
        test_sample = X_test[0]
        print(f"Test sample: {test_sample}")
        print(f"Prediction: {model.predict([test_sample])}")
        # save the model
        pickle.dump(model, open("Trees/model_" + str(value) + ".pkl", "wb"))
        # save the used features
        with open("Trees/used_features.txt", "w") as f:
            f.write(",".join(features[:-1]))

        os.makedirs("LR", exist_ok=True)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        print(f"MSE LinearRegression: {mse}")
        # save the model
        pickle.dump(model, open("LR/model_" + str(value) + ".pkl", "wb"))
        # save the used features
        with open("LR/used_features.txt", "w") as f:
            f.write(",".join(features[:-1]))

if __name__ == '__main__':
    main()