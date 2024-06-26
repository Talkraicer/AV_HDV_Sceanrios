# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.colors as mcolors

# %%
EXP_NAME = "BlockedLane5000"
STOPPING_LANE = 0
METRICS = ["duration", "departDelay", "speed", "timeLoss", "totalDelay"]
vTypes = ["LaneChanger", "all", "AV"]
fontsize = 60


# %%
def sort_df(df):
    split_indices = df.index.str.split('_', expand=True).values.tolist()
    cols = [str(i) for i in range(len(split_indices[0]))]
    splitted_indices_df = pd.DataFrame(split_indices, index=df.index, columns=cols)
    df_joined = df.join(splitted_indices_df)
    df_sorted = df_joined.sort_values(by=cols[::-1])
    df_sorted.drop(columns=cols, inplace=True)
    return df_sorted


# %%
# for loop over vtypes and metrics with tqdm
for STOPPING_LANE in [0, 1, 2]:
    for vType in tqdm(vTypes):
        for metric in tqdm(METRICS):
            df = pd.read_csv(
                f"results_csvs/{EXP_NAME}/stopped_lane_{STOPPING_LANE}/{EXP_NAME}_{metric}_{vType}_stopping_lane_{STOPPING_LANE}.csv",
                index_col=0)
            # sort the rows, but sort by the transposed string
            df = sort_df(df)
            # round the values
            df = df.round(1)
            # plot the table with colors according to the values
            fig, ax = plt.subplots(figsize=(100, 50))

            # Create a colormap
            cmap = plt.cm.coolwarm
            # Set the color for value 0.0 to black
            cmap.set_under(color='black')

            # Mask the DataFrame values that are not 0.0
            df_masked = np.ma.masked_where(df == 0.0, df)

            norm = mcolors.TwoSlopeNorm(vmin=df.min().min(), vcenter=0, vmax=df.max().max())

            # Use the custom colormap
            cax = ax.matshow(df_masked, cmap=cmap, norm=norm)

            fig.colorbar(cax)
            # Add cell values as text on each cell
            for i in range(df.shape[0]):
                for j in range(df.shape[1]):
                    text = ax.text(j, i, df.iloc[i, j],
                                   ha="center", va="center", color="w")

            # highlight min value of each col with yellow background
            for j in range(df.shape[1]):
                min_val = df.iloc[:, j].min()
                for i in range(df.shape[0]):
                    if df.iloc[i, j] == min_val:
                        ax.text(j, i, min_val, ha="center", va="center", color="black", bbox=dict(facecolor='yellow', alpha=0.5))


            # Set the x-axis labels
            ax.set_xticks(np.arange(len(df.columns)))
            ax.set_xticklabels(df.columns)
            # Set the y-axis labels
            ax.set_yticks(np.arange(len(df.index)))
            ax.set_yticklabels(df.index)
            # Rotate the y-axis labels
            plt.setp(ax.get_xticklabels(), rotation=90, ha="right")
            plt.suptitle(f"{EXP_NAME} Average {metric} For {vType} Where The Stopping Lane is {STOPPING_LANE}",
                         fontsize=fontsize)
            plt.savefig(
                f"results_figures/{EXP_NAME}/stopped_lane_{STOPPING_LANE}/{EXP_NAME}_{metric}_{vType}_stopping_lane_{STOPPING_LANE}.png")
            plt.close()

