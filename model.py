"""Train and evaluate direction/magnitude prediction models with a strict
time-ordered (walk-forward) train/test split -- shuffling would leak future
information into training and produce misleadingly good accuracy.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score

from features import FEATURE_COLUMNS, build_features


@dataclass
class TrainResult:
    direction_model: GradientBoostingClassifier
    return_model: GradientBoostingRegressor
    test_df: pd.DataFrame
    test_accuracy: float
    test_auc: float
    test_mae: float
    baseline_accuracy: float
    feature_importances: pd.Series
    latest_prediction: dict


def train_and_evaluate(raw_df: pd.DataFrame, test_fraction: float = 0.2) -> TrainResult:
    feat_df = build_features(raw_df)
    modeling_df = feat_df.dropna(subset=FEATURE_COLUMNS + ["target_direction", "target_return"])

    if len(modeling_df) < 100:
        raise ValueError("Not enough history after feature engineering to train a model (need 100+ rows).")

    split_idx = int(len(modeling_df) * (1 - test_fraction))
    train_df = modeling_df.iloc[:split_idx]
    test_df = modeling_df.iloc[split_idx:]

    X_train, y_dir_train, y_ret_train = (
        train_df[FEATURE_COLUMNS], train_df["target_direction"], train_df["target_return"],
    )
    X_test, y_dir_test, y_ret_test = (
        test_df[FEATURE_COLUMNS], test_df["target_direction"], test_df["target_return"],
    )

    direction_model = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42,
    )
    direction_model.fit(X_train, y_dir_train)

    return_model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42,
    )
    return_model.fit(X_train, y_ret_train)

    dir_pred = direction_model.predict(X_test)
    dir_proba = direction_model.predict_proba(X_test)[:, 1]
    ret_pred = return_model.predict(X_test)

    test_accuracy = accuracy_score(y_dir_test, dir_pred)
    try:
        test_auc = roc_auc_score(y_dir_test, dir_proba)
    except ValueError:
        test_auc = float("nan")  # only one class present in test set
    test_mae = mean_absolute_error(y_ret_test, ret_pred)
    baseline_accuracy = max(y_dir_test.mean(), 1 - y_dir_test.mean())

    result_test_df = test_df.copy()
    result_test_df["pred_direction"] = dir_pred
    result_test_df["pred_prob_up"] = dir_proba
    result_test_df["pred_return"] = ret_pred

    importances = pd.Series(direction_model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)

    latest_row = feat_df.iloc[[-1]]
    latest_features = latest_row[FEATURE_COLUMNS]
    if latest_features.isna().any(axis=None):
        latest_prediction = {"available": False}
    else:
        prob_up = float(direction_model.predict_proba(latest_features)[0, 1])
        pred_return = float(return_model.predict(latest_features)[0])
        latest_prediction = {
            "available": True,
            "as_of": latest_row.index[-1],
            "prob_up": prob_up,
            "predicted_return": pred_return,
            "direction": "Up" if prob_up >= 0.5 else "Down",
            "confidence": max(prob_up, 1 - prob_up),
        }

    return TrainResult(
        direction_model=direction_model,
        return_model=return_model,
        test_df=result_test_df,
        test_accuracy=test_accuracy,
        test_auc=test_auc,
        test_mae=test_mae,
        baseline_accuracy=baseline_accuracy,
        feature_importances=importances,
        latest_prediction=latest_prediction,
    )


def strategy_backtest(test_df: pd.DataFrame) -> pd.DataFrame:
    """Simple long/short backtest: go long when predicted direction is Up, short when Down."""
    bt = test_df.copy()
    position = np.where(bt["pred_direction"] == 1, 1, -1)
    bt["strategy_return"] = position * bt["target_return"]
    bt["strategy_cum_return"] = (1 + bt["strategy_return"]).cumprod() - 1
    bt["buy_hold_cum_return"] = (1 + bt["target_return"]).cumprod() - 1
    return bt
