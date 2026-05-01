import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def is_pareto_efficient_dumb(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i]<c, axis=1)) and np.all(np.any(costs[i+1:]<c, axis=1))
    return is_efficient

# def hypervolume_pareto(costs):
N = 5000

def load_empirical_validation_df(
    split_ratio=0.1,
    seed=42,
):
    # Load all 8 validation result_df
    dfs = []
    for i in range(1,9):
        df = pd.read_csv(f"/mimer/NOBACKUP/groups/naiss2024-22-1336/DCMO_alj/empirical_validation_{i}/results.csv", index_col=0)
        dfs += [df.copy()]
    df = pd.concat(dfs)
    df = df[df["rew_safe"] == 1]
    print(df.shape)
    
    df_means = df.groupby(["controller", "weather", "dist", "speed"])[["rew_eff", "rew_sta"]].mean()
    # print(df_means[["rew_sta", "rew_eff"]])

    moc_mab_results = np.load(f"/mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/bandit_checkpoint_e_s_snapshots/snapshot_t{N}.npz")
    # print(moc_mab_results["mu_hat"].shape)

    df_contexts = pd.read_csv("/mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/stage2/context_cells.csv", index_col=0)
    # print(df_contexts)

    areas_moc_mab = []
    areas_val = []
    for i in range(30):
        [w, d, s] = list(df_contexts.loc[i])
        # print(w,d,s)

        moc_mab = moc_mab_results["mu_hat"][:,i,:]
        # print(moc_mab)
        is_pareto = is_pareto_efficient_dumb(moc_mab)
        # print(moc_mab[is_pareto])
        sorted_front_moc_mab = moc_mab[is_pareto][moc_mab[is_pareto][:, 1].argsort()[::-1]]
        # print(sorted_front_moc_mab)
        xs = [0] + list(sorted_front_moc_mab[:,0])
        ys = [sorted_front_moc_mab[:,1][0]] + list(sorted_front_moc_mab[:,1])
        # print(xs,ys)
        area_moc = np.trapezoid(np.array(ys), x=np.array(xs))
        print(area_moc)
        areas_moc_mab.append(area_moc)

        val = df[(df["weather"] == w) & (df["dist"] == d) & (df["speed"] == s) ].groupby(["controller"])[["rew_eff", "rew_sta"]].mean()
        val = np.array(val)
        is_pareto = is_pareto_efficient_dumb(val)
        # print(val[is_pareto])
        sorted_front_val = val[is_pareto][val[is_pareto][:, 1].argsort()[::-1]]
        # print(sorted_front_val)
        xs = [0] + list(sorted_front_val[:,0])
        ys = [sorted_front_val[:,1][0]] + list(sorted_front_val[:,1])
        # print(xs,ys)
        area_val = np.trapezoid(np.array(ys), x=np.array(xs))
        print(area_val)
        areas_val.append(area_val)


        # print(df[[(df["weather"] == w) & (df["dist"] == d) & (df["speed"] == s)]])
        # val = df[[(df["weather"] == w) & (df["dist"] == d) & (df["speed"] == s)]].groupby(["controller"])[["rew_sta", "rew_eff"]].mean()
        # print(val)
    print(areas_moc_mab, areas_val)

    xs = [i for i in range(30)]
    plt.plot(xs, areas_moc_mab,label="Our approach")
    plt.plot(xs, areas_val, label="Validation")
    plt.title(f"Comparison with {N}")
    plt.xlabel("Contexts")
    plt.ylabel("Area under Pareto front")
    plt.legend()
    plt.savefig(f'plot_{N}.png')
    # # Get dummies from controllers with correct order
    # controllers = list(df["controller"].unique())
    # controllers.sort()
    # cont_dict = {e: controllers.index(e) for e in controllers}
    # df["controller"] = df["controller"].apply(lambda x: cont_dict[x])
    # df = pd.get_dummies(df, columns=["controller"],dtype=float)

    # # Get weather features
    # df_weather = pd.DataFrame(df["weather"].apply(lambda x: Weather[x].value).tolist(), index=df.index, columns=[f"weather_feature_{i}" for i in range(14)]) 
    # df = df.drop(["weather"], axis=1)
    # data = pd.concat([df, df_weather], axis=1)


    # # Randomize order
    # data_size = len(data)
    # data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    # train_sample = int(data_size * (1 - split_ratio))
    # training_data = data.loc[:train_sample]
    # val_data = data.loc[train_sample:]

    # train_dataset = E2E_Dataset(
    #     dataframe=training_data
    # )
    # val_dataset = E2E_Dataset(
    #     dataframe=val_data
    # )
    return df


if __name__ == "__main__":


    print("-- Loading data", flush=True)
    
    load_empirical_validation_df()
    print("-- Training", flush=True)