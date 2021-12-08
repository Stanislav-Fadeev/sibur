import pandas as pd
import numpy as np



import pathlib
from catboost import CatBoostRegressor


from sklearn.metrics import mean_squared_log_error



pd.set_option("display.max_columns", 15)


DATA_DIR = pathlib.Path(".")
DATA_FILE = "sc2021_train_deals.csv"

AGG_COLS = [
    "material_code",
    "company_code",
    "country",
    "region",
    "manager_code",
]
RS = 82736
s = pd.read_csv(DATA_DIR.joinpath(DATA_FILE))
data = pd.read_csv(DATA_DIR.joinpath(DATA_FILE), parse_dates=["month", "date"])
group_ts = data.groupby(AGG_COLS + ["month"])["volume"].sum().unstack(fill_value=0)
tr_data = group_ts.iloc[:, :-1]
val_data = group_ts.iloc[:, -1:]
predictions = val_data.copy()
predictions.iloc[:, :] = np.nan
predictions.iloc[:, 0] = tr_data.iloc[:, -1:]
predictions.ffill(axis=1, inplace=True)





def get_features(df: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    """Calculate features for `month`."""

    start_period = month - pd.offsets.MonthBegin(6)
    end_period = month - pd.offsets.MonthBegin(1)

    df = df.loc[:, :end_period]

    features = pd.DataFrame([], index=df.index)
    features["month"] = month.month
    features[[f"vol_tm{i}" for i in range(6, 0, -1)]] = df.loc[
        :, start_period:end_period
    ].copy()

    rolling = df.rolling(12, axis=1, min_periods=1)
    features = features.join(rolling.mean().iloc[:, -1].rename("last_year_avg"))
    features = features.join(rolling.min().iloc[:, -1].rename("last_year_min"))
    features = features.join(rolling.max().iloc[:, -1].rename("last_year_max"))
    # features = features.join(rolling.median().iloc[:, -1].rename("last_year_med"))

    features["month"] = month.month
    return features



tr_range = pd.date_range("2019-1-01", "2019-6-01", freq="MS")
val_range = pd.date_range("2019-07-01", "2019-12-01", freq="MS")
ts_range = pd.date_range("2020-01-01", "2020-07-1", freq="MS")
full_features = {}

for dataset, dataset_range in zip(["tr", "val", "ts"], [tr_range, val_range, ts_range]):
    dataset_features = []
    for target_month in dataset_range:
        features = get_features(group_ts, target_month)
        features["target"] = group_ts[target_month]
        dataset_features.append(features.reset_index())
    full_features[dataset] = pd.concat(dataset_features, ignore_index=False)
CAT_COLS = [
    "material_code",
    "company_code",
    "manager_code",
    "country",
    "region",
    "month",
]
FTS_COLS = [
    "material_code",
    "company_code",
    "country",
    "region",
    "manager_code",
    "month",
    "vol_tm6",
    "vol_tm5",
    "vol_tm4",
    "vol_tm3",
    "vol_tm2",
    "vol_tm1",
    "last_year_avg",
    "last_year_min",
    "last_year_max",
]
TARGET = "target"


model = CatBoostRegressor(
    iterations=1000,
    loss_function="RMSE",
    learning_rate=(0.077),
    # penalties_coefficient=5,
    l2_leaf_reg=3,
    random_strength=1,
    # leaf_estimation_method="Newton",
    # early_stopping_rounds=75,
    depth=7,
    cat_features=CAT_COLS,
    random_state=1255,
    verbose=0,
    # sampling_frequency=0.8
)


model.fit(
    (full_features["tr"][FTS_COLS]),
    ((full_features["tr"][TARGET]) ** 0.8),
    eval_set=(full_features["val"][FTS_COLS], (full_features["val"][TARGET]) ** 0.8,),
    cat_features=CAT_COLS,
    # plot=(True),
)



tr_preds = model.predict(full_features["tr"][FTS_COLS])
val_preds = model.predict(full_features["val"][FTS_COLS])
ts_preds = model.predict(full_features["ts"][FTS_COLS])
print(
    "Ошибка на тренировочном множестве:",
    f'{np.sqrt(mean_squared_log_error(full_features["tr"][TARGET], tr_preds)):.4f}',
)
print(
    "Ошибка на валидационном множестве:",
    f'{np.sqrt(mean_squared_log_error(full_features["val"][TARGET], val_preds)):.4f}',
)
print(
    "Ошибка на тестовом множестве:",
    f'{np.sqrt(mean_squared_log_error(full_features["ts"][TARGET], ts_preds)):.4f}',
)
# model.save_model("qstestD5.cbm")
