"""Modele simple și robuste (fără deep learning). Toate primesc X cu NaN tratat."""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import xgboost as xgb


def make_model(name: str, seed: int = 42):
    if name == "logreg":   # baseline: regresie logistică puternic regularizată
        return make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    if name == "rf":
        return make_pipeline(SimpleImputer(), RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=50, max_features="sqrt", n_jobs=-1, random_state=seed))
    if name == "lgbm":
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=15, min_child_samples=100,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                                  random_state=seed, verbose=-1)
    if name == "xgb":
        return xgb.XGBClassifier(n_estimators=300, learning_rate=0.02, max_depth=3, min_child_weight=20,
                                 subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, random_state=seed,
                                 n_jobs=4, eval_metric="logloss")
    raise ValueError(name)
