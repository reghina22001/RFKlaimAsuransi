import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Sistem Prediksi Klaim Asuransi BPJS",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling yang lebih baik
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 4rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .metric-card h2,
    .metric-card h3 {
        color: white !important;
    }
    
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .error-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .info-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: #333;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0px 0px;
        gap: 10px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Styling untuk grafik yang lebih rapi */
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .chart-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-top: 1.5rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Kategori fitur lengkap (disimpan sebagai variabel global)
kategori_fitur_lengkap = [
    'Medis',
    'Demografis',
    'Demografis',
    'Finansial',
    'Demografis',
    'Finansial',
    'Finansial',
    'Finansial',
    'Finansial',
    'Finansial',
    'Finansial',
    'Demografis',
    'Finansial',
    'Finansial',
    'Finansial',
    'Administratif',
    'Medis',
    'Administratif',
    'Finansial',
    'Administratif',
    'Administratif',
    'Finansial',
    'Finansial',
    'Medis',
    'Finansial',
    'Finansial',
    'Administratif',
    'Medis',
    'Finansial',
    'Finansial',
    'Finansial',
    'Finansial',
    'Medis',
    'Finansial',
    'Finansial',
    'Finansial',
    'Finansial',
    'Finansial',
    'Finansial',
    'Administratif',
    'Administratif'
]


@st.cache_resource # Menggunakan st.cache_resource untuk model
def load_models():
    """Load trained models and feature columns"""
    try:
        # Sesuaikan path dengan lokasi file model Anda
        clf_model = joblib.load("C_classifier_model.pkl")
        reg_model = joblib.load("C_regressor_model.pkl")
        feature_names = joblib.load("C_feature_columns.pkl")
        
        return clf_model, reg_model, feature_names
    except FileNotFoundError as e:
        st.error(f"❌ File model tidak ditemukan: {str(e)}")
        st.info("💡 Pastikan file model (.pkl) dan feature_columns.pkl berada di direktori yang sama dengan aplikasi ini")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info(f"Detail error: {e}")
        return None, None, None

def prepare_input_data(feature_names, **kwargs):
    """Prepare input data with exact feature names from training"""
    input_data = {}
    
    # Initialize all features with 0.0 for numerical consistency
    for feature in feature_names:
        input_data[feature] = 0.0 
    
    # Map input values to feature names
    feature_mapping = {
        'KELAS_RAWAT': float(kwargs.get('kelas_rawat', 3)),
        'BIRTH_WEIGHT': float(kwargs.get('birth_weight', 0)),
        'SEX': float(kwargs.get('sex', 1)),
        'DISCHARGE_STATUS': float(kwargs.get('discharge_status', 1)),
        'TARIF_INACBG': float(kwargs.get('tarif_inacbg', 0)),
        'TARIF_SUBACUTE': float(kwargs.get('tarif_subacute', 0)),
        'TARIF_CHRONIC': float(kwargs.get('tarif_chronic', 0)),
        'TARIF_SP': float(kwargs.get('tarif_sp', 0)),
        'TARIF_SR': float(kwargs.get('tarif_sr', 0)),
        'TARIF_SI': float(kwargs.get('tarif_si', 0)),
        'TARIF_SD': float(kwargs.get('tarif_sd', 0)),
        'TOTAL_TARIF': float(kwargs.get('total_tarif', 0)),
        'TARIF_POLI_EKS': float(kwargs.get('tarif_poli_eks', 0)),
        'LOS': float(kwargs.get('los', 1)),
        'ICU_INDIKATOR': float(kwargs.get('icu_indikator', 0)),
        'ICU_LOS': float(kwargs.get('icu_los', 0)),
        'VENT_HOUR': float(kwargs.get('vent_hour', 0)),
        'UMUR_TAHUN': float(kwargs.get('umur_tahun', 30)),
        'UMUR_HARI': float(kwargs.get('umur_hari', 0)),
        'VERSI_INACBG': float(kwargs.get('versi_inacbg', 5.8)),
        'VERSI_GROUPER': float(kwargs.get('versi_grouper', 4)),
        'PROSEDUR_NON_BEDAH': float(kwargs.get('prosedur_non_bedah', 0)),
        'PROSEDUR_BEDAH': float(kwargs.get('prosedur_bedah', 0)),
        'KONSULTASI': float(kwargs.get('konsultasi', 0)),
        'TENAGA_AHLI': float(kwargs.get('tenaga_ahli', 0)),
        'KEPERAWATAN': float(kwargs.get('keperawatan', 0)),
        'PENUNJANG': float(kwargs.get('penunjang', 0)),
        'RADIOLOGI': float(kwargs.get('radiologi', 0)),
        'LABORATORIUM': float(kwargs.get('laboratorium', 0)),
        'PELAYANAN_DARAH': float(kwargs.get('pelayanan_darah', 0)),
        'REHABILITASI': float(kwargs.get('rehabilitasi', 0)),
        'KAMAR_AKOMODASI': float(kwargs.get('kamar_akomodasi', 0)),
        'RAWAT_INTENSIF': float(kwargs.get('rawat_intensif', 0)),
        'OBAT': float(kwargs.get('obat', 0)),
        'ALKES': float(kwargs.get('alkes', 0)),
        'BMHP': float(kwargs.get('bmhp', 0)),
        'SEWA_ALAT': float(kwargs.get('sewa_alat', 0)),
        'OBAT_KRONIS': float(kwargs.get('obat_kronis', 0)),
        'OBAT_KEMO': float(kwargs.get('obat_kemo', 0)),
        'LOS_CALCULATED': float(kwargs.get('los_calculated', kwargs.get('los', 1))),
        'AGE_AT_ADMISSION': float(kwargs.get('age_at_admission', kwargs.get('umur_tahun', 30))),
        'TARIF_RS': float(kwargs.get('total_tarif', 0))
    }
    
    # Update input_data with actual values if the feature exists
    for feature_name, value in feature_mapping.items():
        if feature_name in input_data:
            input_data[feature_name] = value
    
    # Create dataframe with exact column order
    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_names]  # Ensure exact order
    
    return input_df

def create_prediction_chart(status_proba):
    """Create a beautiful prediction probability chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=['Ditolak', 'Diterima'],
            y=[status_proba[0], status_proba[1]],
            marker_color=['#ff6b6b', '#4ecdc4'],
            text=[f'{status_proba[0]:.1%}', f'{status_proba[1]:.1%}'],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        xaxis_title="Status Klaim",
        yaxis_title="Probabilitas",
        yaxis=dict(tickformat='.0%'),
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_cost_breakdown_chart(cost_data):
    """Create cost breakdown pie chart"""
    labels = []
    values = []
    
    for label, value in cost_data.items():
        if value > 0:
            labels.append(label)
            values.append(value)
    
    if not values:
        return None
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        textinfo='label+percent',
        textposition='outside'
    )])
    
    fig.update_layout(
        height=400,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_confusion_matrix_chart():
    """Create a Confusion Matrix chart."""
    confusion_matrix_data = [
        [139, 262],
        [33, 1566]
    ]
    
    labels = ["Tidak Diterima", "Diterima"]
    
    fig = px.imshow(confusion_matrix_data,
                    text_auto=True,
                    labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                    x=labels,
                    y=labels,
                    color_continuous_scale="Blues")
    
    fig.update_xaxes(side="bottom")
    fig.update_layout(
    margin=dict(t=50, b=40),
    height=450,
    width=500,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)
    
    return fig

def show_system_info_tab():
    """Display system information tab content"""
    st.markdown("## 📊 Informasi Sistem")
    
    # System overview
    st.markdown("""
    <div class="info-card">
        <h3>🎯 Tentang Sistem Prediksi Klaim BPJS</h3>
        <p>Sistem ini menggunakan teknologi Machine Learning untuk memprediksi status klaim asuransi BPJS 
        dan memberikan estimasi jumlah klaim yang akan diterima. Sistem telah dilatih menggunakan data 
        historis klaim BPJS untuk memberikan prediksi yang akurat dan dapat diandalkan.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Layout grafik yang diperbaiki - menggunakan kolom yang lebih seimbang
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="chart-title">📊 Confusion Matrix Klasifikasi Klaim Asuransi</div>', unsafe_allow_html=True)
        confusion_matrix_chart = create_confusion_matrix_chart()
        st.plotly_chart(confusion_matrix_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-title">🔍 Fitur Penting dalam Prediksi</div>', unsafe_allow_html=True)
        
        # Data feature importance
        feature_importance_data = {
            'Fitur': ['PROSEDUR_NON_BEDAH', 'UMUR_HARI', 'AGE_AT_ADMISSION', 'OBAT', 'UMUR_TAHUN',
                      'BMHP', 'KONSULTASI', 'TARIF_INACBG', 'TOTAL_TARIF', 'ALKES', 'LABORATORIUM',
                      'SEX', 'OBAT_KEMO', 'RADIOLOGI', 'KAMAR_AKOMODASI', 'LOS', 'PROSEDUR_BEDAH',
                      'LOS_CALCULATED', 'PELAYANAN_DARAH', 'KELAS_RAWAT', 'DISCHARGE_STATUS',
                      'SEWA_ALAT', 'REHABILITASI', 'BIRTH_WEIGHT', 'TARIF_SP', 'KEPERAWATAN', 'ICU_LOS',
                      'ICU_INDIKATOR', 'TENAGA_AHLI', 'PENUNJANG', 'OBAT_KRONIS', 'TARIF_SD', 'VENT_HOUR',
                      'TARIF_SR', 'TARIF_SUBACUTE', 'RAWAT_INTENSIF', 'TARIF_CHRONIC', 'TARIF_SI',
                      'VERSI_INACBG', 'TARIF_POLI_EKS', 'VERSI_GROUPER'],

            'Tingkat Kepentingan' : [0.20034084124777327, 0.13956667331674139, 0.13894035171722022,
                                     0.10606254799605024, 0.08171795725401494, 0.06773616682372716,
                                     0.06469253576815329, 0.04083096024140984, 0.03431903559588637,
                                     0.03363531720467421, 0.019058428783495462, 0.01522415372732481,
                                     0.01428317085170302, 0.013488704757400296, 0.01099457024830426,
                                     0.003990101746419787, 0.0034843669664703595, 0.0032572733888876063,
                                     0.003127902774851421, 0.0014817652797593233, 0.0010282016442588113,
                                     0.0008731894476350093, 0.000543638314297092, 0.0004650457936009932,
                                     0.00034680551913388876, 0.000187192708096947, 0.00012899968293918782,
                                     9.322202877336541e-05, 5.350786424991514e-05, 2.4313789643233518e-05,
                                     8.309026835495575e-06, 8.09130944419611e-06, 4.002772295074242e-06,
                                     2.6544085296337334e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

            'Kategori': kategori_fitur_lengkap
        }
        
        # Ambil hanya 15 fitur teratas untuk visualisasi yang lebih rapi
        feature_importance_df = pd.DataFrame(feature_importance_data).sort_values(
            'Tingkat Kepentingan', ascending=True
        ).tail(15)  # Ambil 15 fitur teratas
        
        fig_importance = px.bar(
            feature_importance_df,
            x='Tingkat Kepentingan',
            y='Fitur',
            color='Kategori',
            orientation='h',
            title="",
            color_discrete_map={
                'Finansial': '#3498db',
                'Medis': '#e74c3c',
                'Demografis': '#2ecc71',
                'Administratif': '#f39c12'
            }
        )
        fig_importance.update_layout(
            height=450,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def show_user_input_tab():
    """Display user input tab content"""
    st.markdown("## 📝 Input Data Pengguna")
    
    # Load models
    clf_model, reg_model, feature_names = load_models()
    
    if clf_model is None:
        st.error("❌ Model tidak dapat dimuat. Silakan periksa file model.")
        return
    
    # Create columns for input form
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        st.markdown("### 👤 Informasi Pasien")
        
        # Informasi Dasar Pasien
        with st.expander("👤 Data Dasar Pasien", expanded=True):
            umur_tahun = st.number_input("Umur (Tahun)", min_value=0, max_value=120, value=30, help="Masukkan umur pasien dalam tahun")
            umur_hari = st.number_input("Umur (Hari)", min_value=0, max_value=365*120, value=0, help="Masukkan umur pasien dalam hari (isi 0 jika tidak spesifik, penting untuk bayi)")
            
            sex = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"], help="Pilih jenis kelamin pasien")
            sex_code = 1 if sex == "Laki-laki" else 2
            
            birth_weight = st.number_input("Berat Lahir (gram)", min_value=0, max_value=10000, value=0, help="Kosongkan jika bukan bayi")
            
            kelas_rawat = st.selectbox("Kelas Rawat", [1, 2, 3], index=2, help="Kelas perawatan di rumah sakit")
        
        # Informasi Medis
        with st.expander("🏥 Informasi Medis", expanded=True):
            los = st.number_input("Length of Stay (Hari)", min_value=1, max_value=365, value=1, help="Lama rawat inap dalam hari")
            
            icu_indikator = st.selectbox("Perawatan ICU", ["Tidak", "Ya"], help="Apakah pasien dirawat di ICU?")
            icu_code = 1 if icu_indikator == "Ya" else 0
            
            icu_los = st.number_input("Lama ICU (Hari)", min_value=0, max_value=365, value=0, help="Lama perawatan ICU jika ada")
            
            vent_hour = st.number_input("Jam Ventilator", min_value=0, max_value=8760, value=0, help="Jam penggunaan ventilator")
            
            discharge_status = st.selectbox("Status Pulang", [1, 2, 3, 4, 5], index=0, 
                                             help="1=Sembuh, 2=Rujuk, 3=APS, 4=Meninggal <48jam, 5=Meninggal >48jam")
        
        # Informasi Tarif
        with st.expander("💰 Informasi Tarif", expanded=True):
            tarif_inacbg = st.number_input("Tarif INA-CBG (Rp)", min_value=0, value=500000, step=10000, help="Tarif standar INA-CBG")
            
            total_tarif = st.number_input("Total Tarif RS (Rp)", min_value=0, value=600000, step=10000, help="Total tarif rumah sakit")
        
        # Rincian Biaya Perawatan
        with st.expander("🧾 Rincian Biaya Perawatan", expanded=False):
            col_cost1, col_cost2 = st.columns(2)
            
            with col_cost1:
                prosedur_non_bedah = st.number_input("Prosedur Non-Bedah (Rp)", min_value=0, value=0, step=1000)
                prosedur_bedah = st.number_input("Prosedur Bedah (Rp)", min_value=0, value=0, step=1000)
                konsultasi = st.number_input("Konsultasi (Rp)", min_value=0, value=100000, step=1000)
                tenaga_ahli = st.number_input("Tenaga Ahli (Rp)", min_value=0, value=0, step=1000)
                keperawatan = st.number_input("Keperawatan (Rp)", min_value=0, value=0, step=1000)
                penunjang = st.number_input("Penunjang (Rp)", min_value=0, value=0, step=1000)
                radiologi = st.number_input("Radiologi (Rp)", min_value=0, value=0, step=1000)
                laboratorium = st.number_input("Laboratorium (Rp)", min_value=0, value=75000, step=1000)
            
            with col_cost2:
                pelayanan_darah = st.number_input("Pelayanan Darah (Rp)", min_value=0, value=0, step=1000)
                rehabilitasi = st.number_input("Rehabilitasi (Rp)", min_value=0, value=0, step=1000)
                kamar_akomodasi = st.number_input("Kamar & Akomodasi (Rp)", min_value=0, value=200000, step=1000)
                rawat_intensif = st.number_input("Rawat Intensif (Rp)", min_value=0, value=0, step=1000)
                obat = st.number_input("Obat (Rp)", min_value=0, value=150000, step=1000)
                alkes = st.number_input("Alat Kesehatan (Rp)", min_value=0, value=0, step=1000)
                bmhp = st.number_input("BMHP (Rp)", min_value=0, value=0, step=1000)
                sewa_alat = st.number_input("Sewa Alat (Rp)", min_value=0, value=0, step=1000)
                obat_kronis = st.number_input("Obat Kronis (Rp)", min_value=0, value=0, step=1000)
                obat_kemo = st.number_input("Obat Kemoterapi (Rp)", min_value=0, value=0, step=1000)
        
        # Tombol Prediksi
        st.markdown("---")
        predict_button = st.button("🔮 Prediksi Klaim", type="primary", use_container_width=True)
    
    with col_result:
        st.markdown("### 📊 Hasil Prediksi")
        
        if predict_button:
            # Prepare input data
            input_df = prepare_input_data(
                feature_names,
                kelas_rawat=kelas_rawat,
                birth_weight=birth_weight,
                sex=sex_code,
                discharge_status=discharge_status,
                tarif_inacbg=tarif_inacbg,
                total_tarif=total_tarif,
                los=los,
                icu_indikator=icu_code,
                icu_los=icu_los,
                vent_hour=vent_hour,
                umur_tahun=umur_tahun,
                umur_hari=umur_hari, 
                prosedur_non_bedah=prosedur_non_bedah,
                prosedur_bedah=prosedur_bedah,
                konsultasi=konsultasi,
                tenaga_ahli=tenaga_ahli,
                keperawatan=keperawatan,
                penunjang=penunjang,
                radiologi=radiologi,
                laboratorium=laboratorium,
                pelayanan_darah=pelayanan_darah,
                rehabilitasi=rehabilitasi,
                kamar_akomodasi=kamar_akomodasi,
                rawat_intensif=rawat_intensif,
                obat=obat,
                alkes=alkes,
                bmhp=bmhp,
                sewa_alat=sewa_alat,
                obat_kronis=obat_kronis,
                obat_kemo=obat_kemo
            )
            
            try:
                # Make predictions
                status_pred = clf_model.predict(input_df)[0]
                status_proba = clf_model.predict_proba(input_df)[0]
                
                # Prediction results
                if status_pred == 1:
                    st.markdown("""
                    <div class="success-card">
                        <h3>✅ KLAIM DITERIMA</h3>
                        <p style="font-size: 1.1rem; margin: 0;">Klaim Anda kemungkinan besar akan diterima!</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Predict claim amount if accepted
                    amount_pred = reg_model.predict(input_df)[0]
                    
                    # Display metrics dalam layout yang lebih rapi
                    col_metric1, col_metric2 = st.columns(2)
                    
                    with col_metric1:
                        st.metric("Probabilitas Diterima", f"{status_proba[1]:.1%}", 
                                  delta=f"+{(status_proba[1] - 0.5) * 100:.1f}%")
                    
                    with col_metric2:
                        st.metric("Estimasi Klaim", f"Rp {amount_pred:,.0f}")
                    
                    efficiency = (amount_pred / total_tarif * 100) if total_tarif > 0 else 0
                    st.metric("Efisiensi Klaim", f"{efficiency:.1f}%")
                    
                    # Comparison chart dengan styling yang lebih baik
        
                    st.markdown('<div class="chart-title">📊 Perbandingan Tarif</div>', unsafe_allow_html=True)
                    comparison_data = pd.DataFrame({
                        'Kategori': ['Tarif INA-CBG', 'Total Tarif RS', 'Estimasi Klaim'],
                        'Jumlah': [tarif_inacbg, total_tarif, amount_pred],
                        'Warna': ['#3498db', '#e74c3c', '#2ecc71']
                    })
                    
                    fig_comparison = px.bar(
                        comparison_data, 
                        x='Kategori', 
                        y='Jumlah',
                        color='Warna',
                        color_discrete_map={color: color for color in comparison_data['Warna']},
                        title=""
                    )
                    fig_comparison.update_layout(
                        showlegend=False, 
                        height=300,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                else:
                    st.markdown("""
                    <div class="error-card">
                        <h3>❌ KLAIM DITOLAK</h3>
                        <p style="font-size: 1.1rem; margin: 0;">Klaim Anda kemungkinan akan ditolak.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display metrics
                    col_metric1, col_metric2 = st.columns(2)
                    
                    with col_metric1:
                        st.metric("Probabilitas Ditolak", f"{status_proba[0]:.1%}")
                    
                    with col_metric2:
                        risk_score = status_proba[0] * 100
                        st.metric("Skor Risiko", f"{risk_score:.0f}/100")
                    
                    # Reasons for rejection
                    st.subheader("🔍 Kemungkinan Alasan Penolakan")
                    reasons = []
                    
                    if total_tarif > tarif_inacbg * 1.5:
                        reasons.append("💰 Total tarif RS terlalu tinggi dibanding tarif INA-CBG")
                    if los > 14:
                        reasons.append("⏰ Length of Stay terlalu lama")
                    if icu_code == 1 and icu_los == 0:
                        reasons.append("🏥 Inkonsistensi data ICU (ICU aktif tapi LOS ICU 0)")
                    if prosedur_bedah > tarif_inacbg:
                        reasons.append("🔪 Biaya prosedur bedah terlalu tinggi")
                    
                    if not reasons:
                        reasons.append("📊 Kombinasi faktor risiko lainnya yang terdeteksi oleh model.")
                    
                    for i, reason in enumerate(reasons, 1):
                        st.write(f"{i}. {reason}")
                
                # Probability chart dengan styling yang lebih baik
    
                st.markdown('<div class="chart-title">📈 Grafik Probabilitas Prediksi Klaim</div>', unsafe_allow_html=True)
                prob_chart = create_prediction_chart(status_proba)
                st.plotly_chart(prob_chart, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Summary data dalam format yang lebih rapi
    
                st.markdown('<div class="chart-title">📋 Ringkasan Input</div>', unsafe_allow_html=True)
                
                summary_data = {
                    'Parameter': [
                        'Umur', 'Jenis Kelamin', 'Kelas Rawat', 'Length of Stay', 
                        'ICU', 'Tarif INA-CBG', 'Total Tarif RS', 'Selisih Tarif'
                    ],
                    'Nilai': [
                        f"{umur_tahun} tahun ({umur_hari} hari)",
                        sex,
                        f"Kelas {kelas_rawat}",
                        f"{los} hari",
                        icu_indikator,
                        f"Rp {tarif_inacbg:,.0f}",
                        f"Rp {total_tarif:,.0f}",
                        f"Rp {total_tarif - tarif_inacbg:,.0f}"
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Cost breakdown chart
                cost_data = {
                    'Konsultasi': konsultasi,
                    'Obat': obat,
                    'Laboratorium': laboratorium,
                    'Kamar': kamar_akomodasi,
                    'Prosedur Non-Bedah': prosedur_non_bedah,
                    'Prosedur Bedah': prosedur_bedah,
                    'Radiologi': radiologi,
                    'Lainnya': (tenaga_ahli + keperawatan + penunjang + pelayanan_darah + 
                                rehabilitasi + rawat_intensif + alkes + bmhp + sewa_alat + 
                                obat_kronis + obat_kemo)
                }
                
                cost_chart = create_cost_breakdown_chart(cost_data)
                if cost_chart:
                    with st.container():
                        st.markdown('<div class="chart-title">🧾 Breakdown Biaya Perawatan</div>', unsafe_allow_html=True)
                        st.plotly_chart(cost_chart, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error dalam prediksi: {str(e)}")
                st.info("💡 Silakan periksa input data dan coba lagi")

def main():
    """Main application function"""
    # Header
    st.markdown('<h1 class="main-header">🏥 Sistem Prediksi Klaim Asuransi BPJS</h1>', unsafe_allow_html=True)
    st.markdown("<div style='margin-top: 2.5rem;'></div>", unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2 = st.tabs(["📊 Informasi Sistem", "📝 Input Data Pengguna"])
    
    with tab1:
        show_system_info_tab()
    
    with tab2:
        show_user_input_tab()

if __name__ == "__main__":
    main()

