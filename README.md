# 🧠 Parkinsons – ML App (v8.5)

**Predict tab uses the bundled Production model out-of-the-box** (no upload).  
Replacement is possible **only** via **Retrain → Promote** after demonstrating better metrics on the same data.

## Run
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

- Put your pre-trained model (from the notebook) at `models/best_model.joblib` and its metadata (with `opt_thr`) at `assets/best_model.json`.  
- The app ships with a small baseline so it works immediately; replace with your own before submission.
