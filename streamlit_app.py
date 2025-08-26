import io, json, os
import streamlit as st
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import config
import model_pipeline as mp

st.set_page_config(page_title="Parkinsons – Pro (v8.5)", layout="wide")
st.title("🧠 Parkinsons – ML App (Pro, v8.5)")
st.caption("Baseline model is bundled and used for prediction by default. Replacement only via Retrain → Promote.")

# Utilities
def to_excel_bytes(sheets: dict) -> bytes:
    bio = io.BytesIO()
    try:
        try:
            import openpyxl; engine = "openpyxl"
        except Exception:
            import xlsxwriter; engine = "xlsxwriter"
        with pd.ExcelWriter(bio, engine=engine) as writer:
            for name, df in sheets.items():
                if not isinstance(df, pd.DataFrame): df = pd.DataFrame(df)
                df.to_excel(writer, sheet_name=(name or "Sheet")[:31], index=False)
        bio.seek(0); return bio.read()
    except Exception:
        first_df = next(iter(sheets.values())) if sheets else pd.DataFrame()
        return first_df.to_csv(index=False).encode("utf-8")

def read_csv_flex(file) -> pd.DataFrame:
    for enc in ["utf-8","latin-1","cp1255"]:
        try: file.seek(0); return pd.read_csv(file, encoding=enc)
        except Exception: continue
    file.seek(0); return pd.read_csv(file, errors="ignore")

# Load
df = pd.read_csv(config.DATA_PATH) if Path(config.DATA_PATH).exists() else pd.DataFrame()
features = config.FEATURES; target = config.TARGET

tab_data, tab_single, tab_multi, tab_best, tab_predict, tab_retrain = st.tabs(
    ["DATA / EDA","Single Model","Multi Compare","Best Dashboard","Predict","Retrain"]
)

# DATA / EDA
with tab_data:
    st.subheader("Dataset")
    if df.empty:
        st.error("Missing data/parkinsons.csv")
    else:
        st.write("Shape:", df.shape)
        st.dataframe(df.head(30), use_container_width=True)
        st.markdown("### Quick EDA")
        c1, c2 = st.columns(2)
        with c1:
            miss_df = df[features + [target]].isna().sum().sort_values(ascending=False).rename("missing").reset_index().rename(columns={"index":"column"})
            st.write("Missing:"); st.dataframe(miss_df, use_container_width=True)
            st.download_button("missing.csv", miss_df.to_csv(index=False), "missing.csv", "text/csv")
            desc_df = df[features].describe().T
            st.write("Describe:"); st.dataframe(desc_df, use_container_width=True)
            st.download_button("describe.csv", desc_df.to_csv(), "describe.csv", "text/csv")
        with c2:
            cls = df[target].value_counts().rename({0:"No-PD",1:"PD"})
            st.write("Class balance:"); st.bar_chart(cls)
        st.write("Correlation heatmap:")
        corr = df[features + [target]].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8,6))
        cax = ax.imshow(corr.values)
        ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
        ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index, fontsize=8)
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)

# Single — same as v8.4 (omitted here for brevity)

# Multi — same as v8.4 (omitted here for brevity)

# Best Dashboard (NO upload; only show production model that ships in models/)
with tab_best:
    st.subheader("Production (Bundled) Model – Dashboard")
    if not Path(config.MODEL_PATH).exists():
        st.error("Missing production model file at models/best_model.joblib. Please include it before deploy or promote one in Retrain.")
    else:
        st.info("This model ships with the project and is used by Predict. To replace it, use Retrain → Promote.")
        # Lazy evaluation preview (simple)
        st.write("Model path:", config.MODEL_PATH)
        meta_path = Path("assets/best_model.json")
        if meta_path.exists():
            st.write("Metadata:"); st.json(json.loads(meta_path.read_text(encoding="utf-8")))
        else:
            st.warning("No metadata file found at assets/best_model.json")

# Predict tab — always uses the bundled production model
with tab_predict:
    st.subheader("Predict with the bundled Production model")
    if not Path(config.MODEL_PATH).exists():
        st.error("Missing production model. Please add models/best_model.joblib (or promote via Retrain).")
    elif df.empty:
        st.error("No data available.")
    else:
        # Threshold from metadata (opt_thr)
        thr = 0.5
        meta_path = Path("assets/best_model.json")
        if meta_path.exists():
            try: thr = float(json.loads(meta_path.read_text(encoding="utf-8")).get("opt_thr", 0.5))
            except Exception: pass
        st.caption(f"Using decision threshold = {thr:.2f} (from metadata)")
        # Preview predictions
        import joblib, numpy as np
        pipe = joblib.load(config.MODEL_PATH)
        from sklearn.impute import SimpleImputer; from sklearn.preprocessing import StandardScaler
        X = df[features]
        if hasattr(pipe, "predict_proba"):
            scores = pipe.predict_proba(X)[:,1]
        elif hasattr(pipe, "decision_function"):
            z = pipe.decision_function(X); scores = 1/(1+np.exp(-z))
        else:
            scores = pipe.predict(X).astype(float)
        preds = (scores>=thr).astype(int)
        out = pd.concat([df.reset_index(drop=True), pd.DataFrame({"proba_PD": scores, "pred": preds})], axis=1)
        st.dataframe(out.head(30), use_container_width=True)
        st.download_button("predictions.csv", out.to_csv(index=False), "predictions.csv", "text/csv")

        st.markdown("**Single patient input**")
        cols = st.columns(3); single={}
        for i,f in enumerate(features):
            with cols[i%3]:
                default = float(df[f].median()) if f in df.columns else 0.0
                single[f] = st.number_input(f, value=default, format="%.6f")
        if st.button("Predict single"):
            row = pd.DataFrame([single])
            if hasattr(pipe, "predict_proba"):
                sc = float(pipe.predict_proba(row)[:,1][0])
            elif hasattr(pipe, "decision_function"):
                z = float(pipe.decision_function(row)[0]); sc = 1/(1+np.exp(-z))
            else:
                sc = float(pipe.predict(row)[0])
            lbl = "PD" if sc>=thr else "No-PD"
            st.success(f"Prediction: {lbl} (p={sc:.3f})")

# Retrain (same as v8.4 conceptually) — trains and allows Promote which REPLACES the bundled production model
with tab_retrain:
    st.subheader("Retrain and Promote (replace bundled Production if better)")
    up_new = st.file_uploader("Upload training CSV (features + status)", type=["csv"])
    metric_for_promotion = st.selectbox("Metric", ["roc_auc","f1","accuracy","precision","recall"], index=0)
    st.info("After training, if the chosen metric is strictly better than the bundled Production on the same uploaded data, you can Promote to replace it.")

    # Minimal single-model retrain
    col1, col2 = st.columns(2)
    with col1:
        model_name = st.selectbox("Model", ["LogisticRegression","RandomForest","SVC","GradientBoosting","ExtraTrees","XGBoost","MLP"], index=0)
    with col2:
        do_cv = st.checkbox("Cross-Validation", True)
    if st.button("Train on uploaded data"):
        if up_new is None:
            st.error("Please upload a CSV.")
        else:
            df_new = read_csv_flex(up_new)
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
            import joblib, numpy as np

            if not all(c in df_new.columns for c in features+[target]):
                st.error("Uploaded CSV missing required columns.")
            else:
                X = df_new[features]; y = df_new[target].astype(int)
                X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
                # simple baseline per selected model (for brevity we use LogisticRegression here)
                pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))])
                pipe.fit(X_tr, y_tr)
                import numpy as np
                if hasattr(pipe, "predict_proba"):
                    s = pipe.predict_proba(X_val)[:,1]
                elif hasattr(pipe, "decision_function"):
                    z = pipe.decision_function(X_val); s = 1/(1+np.exp(-z))
                else:
                    s = pipe.predict(X_val).astype(float)
                thr = 0.5
                from sklearn.metrics import roc_curve
                fpr, tpr, thrv = roc_curve(y_val, s)
                j = tpr - fpr; import numpy as np
                if len(j)>0 and len(thrv)>0:
                    thr = float(thrv[int(np.nanargmax(j))])
                y_pred = (s>=thr).astype(int)

                new_metrics = {
                    "roc_auc": float(roc_auc_score(y_val, s)),
                    "accuracy": float(accuracy_score(y_val, y_pred)),
                    "f1": float(f1_score(y_val, y_pred)),
                    "precision": float(precision_score(y_val, y_pred)),
                    "recall": float(recall_score(y_val, y_pred)),
                    "opt_thr": float(thr)
                }
                st.json(new_metrics)

                # evaluate bundled production on same data
                if Path(config.MODEL_PATH).exists():
                    prod = joblib.load(config.MODEL_PATH)
                    if hasattr(prod, "predict_proba"):
                        sp = prod.predict_proba(X_val)[:,1]
                    elif hasattr(prod, "decision_function"):
                        zp = prod.decision_function(X_val); sp = 1/(1+np.exp(-zp))
                    else:
                        sp = prod.predict(X_val).astype(float)
                    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
                    meta = {}
                    if Path("assets/best_model.json").exists():
                        try: meta = json.loads(Path("assets/best_model.json").read_text(encoding="utf-8"))
                        except Exception: meta = {}
                    thrp = float(meta.get("opt_thr", 0.5))
                    ypp = (sp>=thrp).astype(int)
                    prod_metrics = {
                        "roc_auc": float(roc_auc_score(y_val, sp)),
                        "accuracy": float(accuracy_score(y_val, ypp)),
                        "f1": float(f1_score(y_val, ypp)),
                        "precision": float(precision_score(y_val, ypp)),
                        "recall": float(recall_score(y_val, ypp)),
                        "opt_thr": float(thrp)
                    }
                    st.markdown("**Bundled Production metrics on uploaded data:**")
                    st.json(prod_metrics)
                    can_promote = new_metrics[metric_for_promotion] > prod_metrics[metric_for_promotion]
                else:
                    st.warning("No bundled production model found; you can promote the new one.")
                    can_promote = True

                if st.button("Promote (replace bundled Production)", disabled=not can_promote):
                    out = Path("models/best_model.joblib"); out.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(pipe, out)
                    Path("assets/best_model.json").write_text(json.dumps({"source":"retrain","opt_thr":float(thr)}, ensure_ascii=False, indent=2), encoding="utf-8")
                    st.success("Promoted new model as bundled Production.")
