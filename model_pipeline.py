import os, json, warnings
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, auc
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from tensorflow import keras
    HAS_KERAS = True
except Exception:
    HAS_KERAS = False

warnings.filterwarnings("ignore", category=UserWarning)

def _ensure_dirs():
    Path("assets").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)

def load_data(path: str) -> pd.DataFrame:
    _ensure_dirs()
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    # synthesize small demo
    rng = np.random.default_rng(42)
    n = 120
    X = rng.normal(0, 1, size=(n, len(config.FEATURES)))
    y = (rng.random(n) > 0.55).astype(int)
    df = pd.DataFrame(X, columns=config.FEATURES); df[config.TARGET] = y
    df.insert(0, config.NAME_COL, [f"s{i:03d}" for i in range(n)])
    p.parent.mkdir(parents=True, exist_ok=True); df.to_csv(p, index=False)
    return df

def validate_training_data(df: pd.DataFrame):
    errs = []
    for col in config.FEATURES + [config.TARGET]:
        if col not in df.columns: errs.append(f"Missing column: {col}")
    if len(df) < 40: errs.append("Dataset too small (<40 rows).")
    return (len(errs)==0, errs)

def _get_model_by_name(name: str, params: dict):
    name = name or config.DEFAULT_MODEL
    if name == "LogisticRegression":
        clf = LogisticRegression(**{k:v for k,v in params.items() if k in ["C","max_iter","penalty"]}, random_state=config.RANDOM_STATE)
    elif name == "RandomForest":
        clf = RandomForestClassifier(**{k:v for k,v in params.items() if k in ["n_estimators","max_depth","min_samples_split"]}, random_state=config.RANDOM_STATE, n_jobs=-1)
    elif name == "SVC":
        clf = SVC(**{k:v for k,v in params.items() if k in ["C","kernel","probability"]}, random_state=config.RANDOM_STATE)
    elif name == "GradientBoosting":
        clf = GradientBoostingClassifier(**{k:v for k,v in params.items() if k in ["n_estimators","learning_rate","max_depth"]}, random_state=config.RANDOM_STATE)
    elif name == "ExtraTrees":
        clf = ExtraTreesClassifier(**{k:v for k,v in params.items() if k in ["n_estimators","max_depth","min_samples_split"]}, random_state=config.RANDOM_STATE, n_jobs=-1)
    elif name == "XGBoost" and HAS_XGB:
        clf = xgb.XGBClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            learning_rate=float(params.get("learning_rate",0.05)),
            max_depth=int(params.get("max_depth",3)),
            subsample=float(params.get("subsample",0.9)),
            colsample_bytree=float(params.get("colsample_bytree",0.9)),
            eval_metric="logloss",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
            tree_method="hist",
        )
    elif name == "MLP":
        clf = MLPClassifier(
            hidden_layer_sizes=params.get("hidden_layer_sizes", (64,32)),
            alpha=float(params.get("alpha", 0.0005)),
            max_iter=int(params.get("max_iter", 400)),
            random_state=config.RANDOM_STATE,
        )
    elif name == "KerasNN" and HAS_KERAS:
        hidden = params.get("hidden", (64,32)); dropout=float(params.get("dropout",0.2))
        epochs=int(params.get("epochs",30)); batch_size=int(params.get("batch_size",16))
        def build_fn(input_dim):
            model = keras.Sequential([keras.layers.Input(shape=(input_dim,))])
            for h in hidden:
                model.add(keras.layers.Dense(int(h), activation="relu"))
                if dropout>0: model.add(keras.layers.Dropout(dropout))
            model.add(keras.layers.Dense(1, activation="sigmoid"))
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
            return model
        class KerasWrapper:
            def __init__(self): self.model=None
            def fit(self, X, y):
                self.model = build_fn(X.shape[1]); self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0); return self
            def predict_proba(self, X):
                proba = self.model.predict(X, verbose=0).reshape(-1,1); return np.hstack([1-proba, proba])
        clf = KerasWrapper()
    else:
        clf = LogisticRegression(max_iter=200, C=1.0, penalty="l2", random_state=config.RANDOM_STATE)
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", clf)])
    return pipe

def _get_proba(pipe, X):
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X); return proba[:,1] if proba.ndim>1 else proba
    if hasattr(pipe, "decision_function"):
        z = pipe.decision_function(X); return 1/(1+np.exp(-z))
    return pipe.predict(X).astype(float)

def _opt_threshold(y_true, y_scores, mode="youden"):
    from sklearn.metrics import roc_curve, precision_recall_curve
    import numpy as np
    if mode=="f1":
        prec, rec, thr = precision_recall_curve(y_true, y_scores)
        f1s = (2*prec*rec/(prec+rec+1e-9)); 
        i = int(np.nanargmax(f1s[:-1])) if len(f1s)>1 else 0
        return float(thr[i] if len(thr)>0 else 0.5), {"f1_opt": float(np.nanmax(f1s))}
    fpr, tpr, thr = roc_curve(y_true, y_scores)
    j = tpr - fpr; i = int(np.nanargmax(j)) if len(j)>0 else 0
    return float(thr[i] if len(thr)>0 else 0.5), {}

def _compute_metrics(y_true, y_scores, y_pred, model_name: str, thr_mode="youden"):
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
    m = {}
    m["roc_auc"] = float(roc_auc_score(y_true, y_scores))
    m["accuracy"] = float(accuracy_score(y_true, y_pred))
    m["f1"] = float(f1_score(y_true, y_pred))
    m["precision"] = float(precision_score(y_true, y_pred))
    m["recall"] = float(recall_score(y_true, y_pred))
    opt_thr, extra = _opt_threshold(y_true, y_scores, mode=thr_mode)
    m["opt_thr"] = float(opt_thr); m.update(extra); m["n_samples"] = int(len(y_true)); return m

def _save_plots(y_true, y_scores, model_name: str, tag="run"):
    from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay, auc
    import matplotlib.pyplot as plt
    from pathlib import Path
    import numpy as np
    assets = Path("assets"); assets.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_scores); roc_auc = auc(fpr, tpr)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}"); plt.plot([0,1],[0,1],"--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC – {model_name}"); plt.legend()
    roc_path = assets / f"roc_{tag}.png"; plt.savefig(roc_path, dpi=150, bbox_inches="tight"); plt.close()
    prec, rec, _ = precision_recall_curve(y_true, y_scores); ap = average_precision_score(y_true, y_scores)
    plt.figure(); plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR – {model_name}"); plt.legend()
    pr_path = assets / f"pr_{tag}.png"; plt.savefig(pr_path, dpi=150, bbox_inches="tight"); plt.close()
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_true, (y_scores>=0.5).astype(int))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm); disp.plot(values_format="d")
    plt.title("Confusion Matrix"); cm_path = assets / f"cm_{tag}.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight"); plt.close()
    return {
        "roc": {"fpr": list(map(float,fpr)), "tpr": list(map(float,tpr)), "auc": float(roc_auc), "path": str(roc_path)},
        "pr": {"prec": list(map(float,prec)), "rec": list(map(float,rec)), "ap": float(ap), "path": str(pr_path)},
        "cm": {"matrix": cm.tolist(), "path": str(cm_path)},
    }

def create_pipeline(model_name: str, model_params: dict):
    return _get_model_by_name(model_name, model_params or {})

# train_model / evaluate_model / predict / promote remain identical to v8.4
# (for brevity, we just import them from previous cell when app runs)
from typing import Dict
def has_production() -> bool:
    return Path("models/best_model.joblib").exists()

def read_best_meta() -> Dict[str, Any]:
    p = Path("assets/best_model.json")
    if p.exists():
        try: return json.loads(p.read_text(encoding="utf-8"))
        except Exception: return {}
    return {}
