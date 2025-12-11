# Smart Energy Usage Optimizer
Predictive analytics + anomaly detection for household/industrial energy consumption.  
Includes data preprocessing, LSTM-based forecasting, random-forest baselines, isolation-forest anomaly detection, and a Streamlit dashboard.

## What’s included
- `data/energy_data.csv` — sample dataset (uploaded).
- `models/` — pretrained artifacts: `iso_forest.pkl`, `rf_model.pkl`, `scaler.pkl`.
- `src/` — code: `data_generator.py`, `preprocessing.py`, `models.py`, `train.py`, `app.py` (Streamlit).
- `notebooks/energy_analysis.ipynb` — exploratory analysis & model walkthrough.



<img width="1454" height="911" alt="image" src="https://github.com/user-attachments/assets/33054ab7-741e-44f8-a3e2-635e933ea638" />
## Quick start (local)
1. Create virtual env:
```bash
python -m venv .venv
source .venv/bin/activate   # mac/linux
.venv\Scripts\activate      # windows
pip install -r requirements.txt  



