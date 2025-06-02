import numpy as np
import pandas as pd
import math
import pathlib
from functools import partial, reduce

def get_dataset(dataset_type="cross_sectional", drop_na=True, global_only=False, timepoint=0, dataset_path=None):
    index_col = ["Subject", "Site", "Time"]
    if dataset_type != "cross_sectional":
        index_col.pop(-1)
    df = get_combat_dataset(
        dataset_path, index_col=index_col, drop_nan=drop_na, global_only=global_only
    )
    available_datasets = {
        "cross_sectional": get_cross_sectional_dataset,
        "longitudinal": get_longitudinal_dataset,
        "delta": get_delta_features_df,
    }
    if dataset_type not in available_datasets.keys():
        raise ValueError(f"Error, please choose type from {available_datasets.keys()}")
    return available_datasets[dataset_type](df, timepoint)


def get_combat_dataset(dataset_path, index_col, drop_nan, global_only):
    col_mapping = {
        "data_combat.Subject": "Subject",
        "data_combat.Time": "Time",
        "data_combat.Site": "Site",
    }
    fun_df = get_data_by_parent_dir(
        dataset_path / "fMRI", "fun", col_mapping, global_only
    )
    struct_df = get_data_by_parent_dir(
        dataset_path / "DTI", "str", col_mapping, global_only
    )
    cognitive_df = get_data_by_parent_dir(
        dataset_path / "cognitive", "cog", col_mapping, global_only
    )


    df = pd.merge(
        left=fun_df, right=struct_df, on=("Subject", "Site", "Time"), how="outer"
    )

    df["Age"] = df["Age_x"].combine(df["Age_y"], lambda x,y: x if pd.isna(y) else y)
    df = df.drop(columns=["Age_x","Age_y","Group_x","Group_y"])
    
    # we don't have cognitive data at t=1
    # for code compatiblility, we make an outer merge that duplicates the values for t=0 and t=1
    # we drop columns for t=1 in get_longitudinal_dataset() later
    # In case t=1 data becomes available, we don't drop the Time column, merge on (Subject, Site, Time), and update get_longitudinal_dataset()
    cognitive_df = cognitive_df.drop(columns=["Age","Time"])
    cognitive_df = cognitive_df.replace(" ", np.nan).dropna().astype(int)
    df = pd.merge(
        left=df, right=cognitive_df, on=("Subject", "Site"), how="outer"
    )

    df = df.set_index(index_col)
    if drop_nan:
        df = df.drop(get_nan_index(df), axis=0)

    return df


def get_data_by_parent_dir(root_path, col_prefix, col_mapping, global_only):
    label_cols = ("Subject", "Time", "Site", "Age", "Group")
    dfs = sorted(root_path.glob("**/*.csv"),key=lambda p: str.casefold(str(p)))
    _GLOBAL_DATA_FILES = [
        "fMRI_ID_CV_HM.csv",
        "Harmzd_fMRI_global.csv",
        "Harmzd_fMRI_MI.csv",
        "DTI_ID_CV.csv",
        "Harmzd_DTI_global.csv",
        "Harmzd_DTI_MI.csv",
        "Harmzd_DTI_density_intensity.csv",  # Density/intensity
        "Harmzd_fMRI_density_intensity.csv",
        "Harmzd_fMRI_Sync.csv",  # Sync data
        "Harmzd_DTI_Sync.csv",
        "cognitive.csv"
    ]
    if global_only:
        dfs = [df for df in dfs if pathlib.Path(df).name in _GLOBAL_DATA_FILES]
    dfs = list(
        map(
            partial(load_and_rename_index, col_mapping=col_mapping),
            dfs,
        )
    )
    df = reduce(
        lambda left, right: pd.merge(
            left, right, how="outer", on=("Subject", "Site", "Time")
        ),
        dfs,
    )
    df = drop_unwanted_columns(df)
    df = rename_combat_cols(df)
    col_name_map = {
        col: f"{col_prefix}_{col}" for col in df.columns if col not in label_cols
    }
    df = df.rename(col_name_map, axis=1)
    return df


def load_and_rename_index(df_path, col_mapping):
    df = pd.read_csv(df_path)
    for k, v in col_mapping.items():
        if k in df:
            df[v] = df[k]
            df = df.drop(columns=[k])
    if "Unnamed: 0" in df:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def drop_unwanted_columns(df):
    df = df.drop(columns=df.filter(like="delta2hat").columns)
    df = df.drop(columns=df.filter(like="gammahat").columns)
    df = df.drop(columns=df.filter(like="gammastarhat").columns)
    df = df.drop(columns=df.filter(like="delta2starhat").columns)
    return df


def rename_combat_cols(df):
    _sync_col_found = False
    name_map = {}
    for col in df.columns:
        if "combat" in col:
            if "Sync" in col:
                if _sync_col_found:
                    raise ValueError("Multiple columns found with name Sync.combat!")
                name_map[col] = "Sync"  # Unique named col
                _sync_col_found = True
            else:
                new_name = col.split(".")[1]
                name_map[col] = new_name
    df = df.rename(name_map, axis=1)
    return df


def get_nan_index(df):
    return df.loc[df.isna().T.any()].index.to_list()


def get_delta_features_df(df, *args):
    df = df.copy()
    df = get_df_with_only_2_visits(df)
    non_delta_cols = ["Group", "fun_HM", "Age", "Time"]
    t0, t1 = get_t0_t1(df)
    non_delta_features = get_non_delta_features(t0, t1)
    df = t1.drop(columns=non_delta_cols) - t0.drop(columns=non_delta_cols)
    df = df.merge(non_delta_features, left_index=True, right_index=True)
    return df


def get_non_delta_features(t0, t1):
    non_delta_cols = [
        "Group",
        "fun_HM",
        "Age",
    ]
    non_delta_features = t0[non_delta_cols]
    non_delta_features = non_delta_features.merge(
        t1["fun_HM"].rename("fun_HM_1"), right_index=True, left_index=True
    )
    non_delta_features = non_delta_features.merge(
        t1["Age"].rename("Age_1"), right_index=True, left_index=True
    )
    return non_delta_features


def get_df_with_only_2_visits(df):
    return df.loc[(df.reset_index().groupby(["Subject", "Site"]).count()["Time"] >= 2)]


def get_t0_t1(df):
    t0 = df.loc[df["Time"] == 0]
    t1 = df.loc[df["Time"] == 1]
    return t0, t1


def get_train_test_split(df, test_split=None, test_idx=None):
    if test_idx is None and test_split is None:
        raise ValueError("Must pass test split or test idx!")
    unique_subjects = df.index.unique()
    if test_split is not None:
        test = np.random.permutation(math.ceil(test_split * len(unique_subjects)))
        train = [idx for idx in range(len(unique_subjects)) if idx not in test]
        test_idx = unique_subjects[test]
        train_idx = unique_subjects[train]
    else:  # test_idx not none
        if not isinstance(test_idx, list):
            test_idx = [test_idx]
        test_idx = df.iloc[test_idx].index
        train_idx = [idx for idx in df.index if idx not in test_idx]
    return df.loc[train_idx], df.loc[test_idx]


def get_longitudinal_dataset(df, *args):
    df = df.copy()
    t0, t1 = get_t0_t1(df)
    t0 = t0.drop(columns=["Time"])
    t1 = t1.drop(columns=["Time", "Group"])  # We use group label at t0
    df = t0.merge(t1, left_index=True, right_index=True, suffixes=("_0", "_1"))

    # drop cognitive columns with t_1 because we don't actually have the data
    cognitive_columns_t1 = df.columns[df.columns.str.startswith('cog') & df.columns.str.endswith('_1')]
    df = df.drop(columns=cognitive_columns_t1)

    # older versions of pandas the outer merge opreation did not sort the dataframe by index
    # since the environments on Compute Canada and ARC use different versions
    # we sort the dataframe to get consistent results
    df = df.sort_index()

    return df


def get_X_y_split(df_train, df_test):
    y_train, y_test = df_train["Group"], df_test["Group"]
    X_train, X_test = df_train.drop(columns=["Group"]), df_test.drop(columns=["Group"])
    return X_train, X_test, y_train, y_test


def remove_unlabelled_subjects(df):
    nan_groups = df.loc[df["Group"].isna()].index
    unlabelled_subjects = []
    for idx in nan_groups:
        if len(df.loc[idx]) < 2 or isinstance(df.loc[idx], pd.Series):
            unlabelled_subjects.append(idx)
        else:
            group_label = df.loc[idx].loc[~df.loc[idx]["Group"].isna()]["Group"].item()
            df.loc[idx]["Group"].fillna(group_label, inplace=True)
    df = df.drop(index=unlabelled_subjects)
    return df

def get_cross_sectional_dataset(df, timepoint):
    df = df.reset_index("Time")
    df = df[df["Time"] == timepoint].drop(columns=["Time"]).dropna(subset=["Group"])

    if df.isna().any(axis=None):
        df = handle_missing_data(df)
    return df

def handle_missing_data(df):
    raise NotImplementedError("Data contains missing values. Write the handling strategy in the 'handle_missing_data' function")
