import os
import pandas as pd
import numpy as np
import joblib
import shap
import requests

# ============================================================
# MODEL YÜKLEME
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_model_and_data():
    model = joblib.load(os.path.join(BASE_DIR, "models", "intelliprice_model.pkl"))
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "intelliprice_featured.csv"))

    features = [
        'Monthly Premium Auto', 'Months Since Last Claim', 'Months Since Policy Inception',
        'Number of Open Complaints', 'Number of Policies', 'Income', 'city_risk_score',
        'parts_inflation_index', 'no_claim_score', 'income_risk_ratio_log',
        'location_score', 'Coverage', 'Vehicle Class', 'Vehicle Size',
        'EmploymentStatus', 'Marital Status', 'Gender'
    ]

    cat_cols = ['Coverage', 'Vehicle Class', 'Vehicle Size',
                'EmploymentStatus', 'Marital Status', 'Gender']

    X = df[features].copy()
    X[cat_cols] = X[cat_cols].astype('category')
    explainer = shap.TreeExplainer(model)

    return model, X, explainer, features, cat_cols

# ============================================================
# CANLI KUR
# ============================================================
def get_kur():
    try:
        r = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=5)
        return r.json()["rates"]["TRY"]
    except:
        return 38.5

# ============================================================
# RİSK ETİKETİ
# ============================================================
def get_risk_label(tahmin_usd):
    if tahmin_usd > 600:
        return "🔴 Yüksek Risk Müşterisi!", "error"
    elif tahmin_usd > 300:
        return "🟡 Orta Risk Profili", "warning"
    else:
        return "🟢 Düşük Risk Profili", "success"

# ============================================================
# HASARSIZLIK SKORU
# ============================================================
def get_no_claim_score(months):
    if months <= 6:
        return 1
    elif months <= 12:
        return 2
    elif months <= 24:
        return 3
    else:
        return 4

# ============================================================
# TAHMİN
# ============================================================
def predict(model, X_ref, explainer, features, cat_cols, user_inputs):
    ornek_musteri = X_ref.iloc[10].copy()

    ornek_musteri['location_score']                = user_inputs['loc_score']
    ornek_musteri['city_risk_score']               = user_inputs['city_risk']
    ornek_musteri['Monthly Premium Auto']          = user_inputs['premium']
    ornek_musteri['Income']                        = user_inputs['income']
    ornek_musteri['Vehicle Class']                 = user_inputs['vehicle_class']
    ornek_musteri['parts_inflation_index']         = user_inputs['parts_inflation']
    ornek_musteri['Months Since Last Claim']       = user_inputs['months_since_claim']
    ornek_musteri['Months Since Policy Inception'] = user_inputs['months_since_inception']
    ornek_musteri['Number of Open Complaints']     = user_inputs['num_complaints']
    ornek_musteri['Number of Policies']            = user_inputs['num_policies']
    ornek_musteri['no_claim_score']                = get_no_claim_score(user_inputs['months_since_claim'])

    # Gelir risk oranı
    income = user_inputs['income']
    premium = user_inputs['premium']
    ratio = premium / (income / 12) if income > 0 else premium
    ornek_musteri['income_risk_ratio_log'] = np.log1p(ratio)

    input_df = pd.DataFrame([ornek_musteri])
    input_df[cat_cols] = input_df[cat_cols].astype('category')

    # Tahmin
    tahmin_usd = model.predict(input_df)[0]
    kur = get_kur()
    tahmin_tl = tahmin_usd * kur
    satis_fiyati_tl = tahmin_tl * 1.20

    # SHAP
    shap_values = explainer.shap_values(input_df)
    shap_explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=input_df.iloc[0],
        feature_names=features
    )

    return {
        "tahmin_usd": tahmin_usd,
        "kur": kur,
        "tahmin_tl": tahmin_tl,
        "satis_fiyati_tl": satis_fiyati_tl,
        "shap_explanation": shap_explanation,
        "parts_inflation": user_inputs['parts_inflation'],
        "months_since_claim": user_inputs['months_since_claim']
    }