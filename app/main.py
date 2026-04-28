import streamlit as st
import matplotlib.pyplot as plt
import shap
from predictor import load_model_and_data, get_kur, get_risk_label, predict

# ============================================================
# SAYFA AYARLARI
# ============================================================
st.set_page_config(
    page_title="IntelliPrice | Kasko Fiyatlama",
    page_icon="🛡️",
    layout="wide"
)

# ============================================================
# MODEL YÜKLEME (cache)
# ============================================================
@st.cache_resource
def cached_load():
    return load_model_and_data()

@st.cache_data(ttl=3600)
def cached_kur():
    return get_kur()

model, X_ref, explainer, features, cat_cols = cached_load()
kur = cached_kur()

# ============================================================
# BAŞLIK
# ============================================================
st.title("🛡️ IntelliPrice: Yapay Zeka Tabanlı Dinamik Kasko Fiyatlama")
st.markdown("**LightGBM & Tweedie Regresyonu** ile tahmini hasar maliyeti hesaplanır. SHAP ile karar şeffaf biçimde açıklanır.")
st.sidebar.caption(f"💱 Anlık Kur: 1 USD = {kur:.2f} ₺")

# ============================================================
# SOL PANEL
# ============================================================
st.sidebar.header("📋 Müşteri & Araç Bilgileri")

st.sidebar.subheader("🏙️ Bölge & Konum")
location_name = st.sidebar.selectbox(
    "Yaşadığı Şehir / Bölge",
    ["Kırsal (Tokat vb.)", "Banliyö (İzmir vb.)", "Metropol (İstanbul)"]
)
location_map  = {"Kırsal (Tokat vb.)": 1, "Banliyö (İzmir vb.)": 2, "Metropol (İstanbul)": 3}
city_risk_map = {"Kırsal (Tokat vb.)": 2, "Banliyö (İzmir vb.)": 4, "Metropol (İstanbul)": 5}
loc_score  = location_map[location_name]
city_risk  = city_risk_map[location_name]

st.sidebar.subheader("💰 Finansal Bilgiler")
premium = st.sidebar.slider("Aylık Prim (USD)", 50, 300, 100)
income  = st.sidebar.number_input("Yıllık Gelir (USD)", 0, 150000, 50000)

st.sidebar.subheader("🚗 Araç Bilgileri")
vehicle_class = st.sidebar.selectbox(
    "Araç Sınıfı",
    ["Two-Door Car", "Four-Door Car", "SUV", "Luxury SUV", "Sports Car", "Luxury Car"]
)
luxury_vehicles = ['Luxury SUV', 'Luxury Car', 'Sports Car']
parts_inflation = 1.5 if vehicle_class in luxury_vehicles else 1.0

st.sidebar.subheader("📅 Poliçe Geçmişi")
months_since_claim      = st.sidebar.slider("Son Hasardan Geçen Ay", 0, 35, 12)
months_since_inception  = st.sidebar.slider("Poliçe Başlangıcından Geçen Ay", 0, 99, 24)
num_complaints          = st.sidebar.slider("Açık Şikayet Sayısı", 0, 5, 0)
num_policies            = st.sidebar.slider("Poliçe Sayısı", 1, 9, 1)

hesapla = st.sidebar.button("🚀 Fiyatı Hesapla ve Analiz Et", use_container_width=True)

# ============================================================
# ANA EKRAN
# ============================================================
if hesapla:
    with st.spinner("Yapay Zeka aktüeryal riski hesaplıyor..."):

        user_inputs = {
            "loc_score": loc_score,
            "city_risk": city_risk,
            "premium": premium,
            "income": income,
            "vehicle_class": vehicle_class,
            "parts_inflation": parts_inflation,
            "months_since_claim": months_since_claim,
            "months_since_inception": months_since_inception,
            "num_complaints": num_complaints,
            "num_policies": num_policies
        }

        sonuc = predict(model, X_ref, explainer, features, cat_cols, user_inputs)

        # METRİKLER
        st.subheader("💡 Poliçe Fiyat Teklifi")
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Tahmini Hasar Maliyeti (Safi Prim)",
            f"₺ {sonuc['tahmin_tl']:,.2f}",
            f"≈ {sonuc['tahmin_usd']:.2f} USD"
        )
        col2.metric(
            "Önerilen Kasko Satış Fiyatı",
            f"₺ {sonuc['satis_fiyati_tl']:,.2f}",
            "+%20 Kar/Gider Marjı"
        )

        risk_label, risk_type = get_risk_label(sonuc['tahmin_usd'])
        if risk_type == "error":
            col3.error(risk_label)
        elif risk_type == "warning":
            col3.warning(risk_label)
        else:
            col3.success(risk_label)

        if sonuc['months_since_claim'] > 24:
            st.success("✅ Bu müşteri hasarsızlık indirimine hak kazanmıştır.")
        if sonuc['parts_inflation'] == 1.5:
            st.warning("⚠️ Lüks/Spor araç: Yedek parça maliyet çarpanı aktif (x1.5)")

        st.divider()

        # SHAP
        st.subheader("🔍 Yapay Zeka Karar Açıklaması (SHAP)")
        st.markdown("Modelin bu müşteriye **neden bu fiyatı verdiğini** şeffaf biçimde gösterir.")

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(sonuc['shap_explanation'], show=False)
        st.pyplot(fig)
        plt.close()

else:
    st.info("👈 Sol panelden müşteri bilgilerini girip **'Fiyatı Hesapla'** butonuna basın.")