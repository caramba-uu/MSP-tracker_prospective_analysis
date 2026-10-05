import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from nonconformist.icp import IcpClassifier
from nonconformist.nc import ClassifierNc, MarginErrFunc
from nonconformist.base import ClassifierAdapter


# NOTE: the saved ICP model references this function by name, so it has to be
# importable into the notebook namespace (``from model_utils import lambda_fuction``)
# before ``joblib.load`` is called.
def lambda_fuction(x):
    return x[1]


def build_icp(min_samples_split=2,
              n_estimators=150,
              min_samples_leaf=5,
              criterion="gini",
              class_weight="balanced",
              max_depth=None):
    clf = RandomForestClassifier(min_samples_split=min_samples_split,
                                 n_estimators=n_estimators,
                                 min_samples_leaf=min_samples_leaf,
                                 criterion=criterion,
                                 class_weight=class_weight,
                                 max_depth=max_depth,
                                 n_jobs=-1)
    icp = IcpClassifier(ClassifierNc(ClassifierAdapter(clf),
                                     MarginErrFunc()), condition=lambda_fuction)
    return icp


def train_icp(icp, train_df, calib_df, features, y_label, model_path=None, rf_path=None):
    icp.fit(train_df[features], train_df[y_label])
    icp.calibrate(calib_df[features].values, calib_df[y_label].values.ravel())

    if model_path is not None:
        joblib.dump(icp, model_path)
    if rf_path is not None:
        joblib.dump(icp.nc_function.model.model, rf_path)
    return icp


def cp_predictions_to_labels(pred_bool):
    """Convert conformal prediction sets to labels.
    0 = RRMS only, 1 = SPMS only, 2 = both labels, 3 = empty set."""
    pred_bool = np.asarray(pred_bool, dtype=bool)
    result_pred = np.full(len(pred_bool), 3)
    result_pred[pred_bool[:, 0] & ~pred_bool[:, 1]] = 0
    result_pred[~pred_bool[:, 0] & pred_bool[:, 1]] = 1
    result_pred[pred_bool[:, 0] & pred_bool[:, 1]] = 2
    return result_pred


def predict_labels(icp, input_df, x_columns, significance=0.07):
    pred_bool = icp.predict(input_df[x_columns].values, significance)
    return cp_predictions_to_labels(pred_bool)
