import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="MobilAI - Sistem Klasifikasi Harga Mobil",
    page_icon="🤖🚗",
    layout="wide"
)

# CSS custom yang lebih clean dan modern
st.markdown("""
<style>
    /* Reset dan Base Styles */
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Header Styles */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #00B4DB, #0083B0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .sub-header {
        text-align: center;
        color: #2D3748;
        margin-bottom: 3rem;
        font-size: 1.3rem;
        font-weight: 400;
        line-height: 1.6;
        padding: 0 2rem;
    }
    
    /* ML Badge */
    .ml-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-left: 1rem;
        vertical-align: middle;
    }
    
    /* Search Section */
    .search-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 2.5rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .search-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2D3748;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Car Card Styles */
    .car-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        border: 1px solid #E2E8F0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .car-card:hover {
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
        transform: translateY(-5px) scale(1.01);
    }
    
    .car-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 5px;
        height: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .car-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 2px solid #F1F5F9;
    }
    
    .car-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1A202C;
        margin: 0;
        line-height: 1.3;
    }
    
    .price-section {
        text-align: right;
    }
    
    .price-label {
        font-size: 0.9rem;
        color: #718096;
        font-weight: 600;
        margin-bottom: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .price-tag {
        font-size: 2.2rem;
        font-weight: 900;
        color: #E53E3E;
        line-height: 1;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    /* Category Badge */
    .category-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .category-economy {
        background: linear-gradient(135deg, #38B2AC, #319795);
        color: white;
    }
    
    .category-medium {
        background: linear-gradient(135deg, #4299E1, #3182CE);
        color: white;
    }
    
    .category-premium {
        background: linear-gradient(135deg, #9F7AEA, #805AD5);
        color: white;
    }
    
    .category-luxury {
        background: linear-gradient(135deg, #ED8936, #DD6B20);
        color: white;
    }
    
    /* Specification Grid */
    .specs-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
    }
    
    .spec-section {
        background: linear-gradient(135deg, #F7FAFC 0%, #EDF2F7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #E2E8F0;
        transition: transform 0.3s ease;
    }
    
    .spec-section:hover {
        transform: translateY(-2px);
    }
    
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #2D3748;
        margin-bottom: 1.25rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #CBD5E0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .spec-list {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }
    
    .spec-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 0;
    }
    
    .spec-item:not(:last-child) {
        border-bottom: 1px solid #E2E8F0;
    }
    
    .spec-label {
        color: #4A5568;
        font-weight: 600;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .spec-value {
        color: #1A202C;
        font-weight: 700;
        font-size: 1rem;
    }
    
    /* ML Section Styles */
    .ml-section {
        background: white;
        padding: 2.5rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
    }
    
    .ml-header {
        font-size: 2rem;
        font-weight: 800;
        color: #2D3748;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #F7FAFC 0%, #EDF2F7 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #E2E8F0;
    }
    
    .prediction-result {
        font-size: 1.8rem;
        font-weight: 800;
        text-align: center;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Stats Section */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .stats-card {
        background: white;
        padding: 1.8rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        border: 1px solid #E2E8F0;
        transition: transform 0.3s ease;
    }
    
    .stats-card:hover {
        transform: translateY(-5px);
    }
    
    .stats-number {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2D3748;
        margin-bottom: 0.5rem;
    }
    
    .stats-label {
        font-size: 0.95rem;
        color: #718096;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Button Styles */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* ML Button */
    .ml-button {
        background: linear-gradient(135deg, #00B4DB, #0083B0) !important;
    }
    
    .ml-button:hover {
        background: linear-gradient(135deg, #0099C3, #007194) !important;
    }
    
    /* Input Styles */
    .stTextInput input, .stNumberInput input {
        border-radius: 12px;
        border: 2px solid #E2E8F0;
        padding: 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        background: white;
    }
    
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
    }
    
    /* Tab Styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #F7FAFC 0%, #EDF2F7 100%);
        border: 2px solid #E2E8F0;
        font-weight: 600;
        color: #4A5568;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-color: #667eea !important;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .specs-grid {
            grid-template-columns: 1fr;
            gap: 1rem;
        }
        
        .stats-container {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .car-header {
            flex-direction: column;
            gap: 1rem;
        }
        
        .price-section {
            text-align: left;
        }
        
        .search-container {
            padding: 1.5rem;
        }
        
        .ml-section {
            padding: 1.5rem;
        }
        
        .main-header {
            font-size: 2.2rem;
        }
        
        .sub-header {
            font-size: 1rem;
            margin-bottom: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==================== FUNGSI UTAMA ====================

# Fungsi untuk membersihkan data
def clean_numeric(value):
    if pd.isna(value):
        return 0
    if isinstance(value, str):
        value = value.replace('$', '').replace(',', '').replace('N/A', '0').replace('?', '')
        if '-' in value and 'cc' not in value and 'km' not in value:
            value = value.split('-')[0]
        if '/' in value:
            value = value.split('/')[0]
        
        cleaned = ''.join(char for char in value if char.isdigit() or char == '.' or char == '-')
        value = cleaned if cleaned else '0'
    
    try:
        return float(value) if value else 0
    except:
        return 0

def clean_price(price):
    if pd.isna(price):
        return 0
    if isinstance(price, str):
        price = price.replace('$', '').replace(',', '').replace(' ', '')
        if '-' in price:
            price = price.split('-')[0]
        if '/' in price:
            price = price.split('/')[0]
        try:
            return float(price)
        except:
            return 0
    return price

def clean_cc(value):
    if pd.isna(value):
        return 0
    if isinstance(value, str):
        value = value.strip()
        if '/' in value:
            value = value.split('/')[0].strip()
        
        cleaned_value = ''
        for char in value:
            if char.isdigit() or char == '.':
                cleaned_value += char
            elif char in ['c', 'k', 'w', 'h', ' ']:
                continue
            else:
                break
        
        if cleaned_value:
            try:
                return float(cleaned_value)
            except ValueError:
                return 0
        else:
            return 0
    return value

def clean_performance(value):
    if pd.isna(value):
        return 0
    if isinstance(value, str):
        value = value.strip()
        if 'sec' in value:
            value = value.replace('sec', '').strip()
        if '/' in value:
            value = value.split('/')[0].strip()
        if '. ' in value:
            value = value.replace('. ', '.')
        cleaned = ''.join(char for char in value if char.isdigit() or char == '.' or char == '-')
        value = cleaned if cleaned else '0'
    
    try:
        return float(value) if value else 0
    except:
        return 0

# Load data dengan error handling untuk deploy
@st.cache_data(show_spinner=False)
def load_data():
    try:
        # Coba berbagai kemungkinan path file
        possible_paths = [
            'Cars Datasets 2025.csv',
            './Cars Datasets 2025.csv',
            'PREDIKSI_HARGA_MOBIL/Cars Datasets 2025.csv'
        ]
        
        df = None
        file_path = None
        
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    file_path = path
                    break
            except:
                continue
        
        if file_path is None:
            st.error("❌ File dataset tidak ditemukan! Pastikan 'Cars Datasets 2025.csv' ada di repository.")
            return None
        
        # Coba berbagai encoding
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'windows-1252', 'cp1252']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                continue
        
        if df is None:
            # Fallback: baca dengan error handling
            try:
                df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
            except Exception as e:
                st.error(f"❌ Gagal memuat data: {e}")
                return None
        
        # Validasi kolom yang diperlukan
        required_columns = ['Company Names', 'Cars Names', 'Cars Prices', 'HorsePower', 'Torque', 
                          'Total Speed', 'CC/Battery Capacity', 'Performance(0 - 100 )KM/H', 
                          'Fuel Types', 'Seats', 'Engines']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"❌ Kolom yang hilang: {missing_columns}")
            return None
        
        # Bersihkan data
        df['Cars Prices'] = df['Cars Prices'].apply(clean_price)
        df['HorsePower'] = df['HorsePower'].apply(clean_numeric)
        df['Torque'] = df['Torque'].apply(clean_numeric)
        df['Total Speed'] = df['Total Speed'].apply(clean_numeric)
        df['CC/Battery Capacity'] = df['CC/Battery Capacity'].apply(clean_cc)
        df['Performance(0 - 100 )KM/H'] = df['Performance(0 - 100 )KM/H'].apply(clean_performance)
        
        # Handle missing values
        df = df.fillna({
            'Company Names': 'Unknown',
            'Cars Names': 'Unknown',
            'Fuel Types': 'Unknown',
            'Seats': 'Unknown',
            'Engines': 'Unknown'
        })
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error dalam memuat data: {str(e)}")
        return None

# ==================== MACHINE LEARNING FUNCTIONS ====================

def classify_price_range(price):
    """Klasifikasi harga ke dalam kategori"""
    if price < 30000:
        return "Ekonomi (<$30K)"
    elif price < 60000:
        return "Menengah ($30K-$60K)"
    elif price < 100000:
        return "Premium ($60K-$100K)"
    else:
        return "Luxury (>$100K)"

def get_category_class(category):
    """Get CSS class for category badge"""
    if "Ekonomi" in category:
        return "category-economy"
    elif "Menengah" in category:
        return "category-medium"
    elif "Premium" in category:
        return "category-premium"
    elif "Luxury" in category:
        return "category-luxury"
    return ""

@st.cache_resource(show_spinner=False)
def train_ml_model(df):
    """Train Random Forest Classifier untuk klasifikasi harga"""
    try:
        # Buat copy dataframe
        df_ml = df.copy()
        
        # Tambahkan kategori harga
        df_ml['Price_Category'] = df_ml['Cars Prices'].apply(classify_price_range)
        
        # Encoding fitur kategorikal
        le_company = LabelEncoder()
        le_fuel = LabelEncoder()
        
        df_ml['Company_Encoded'] = le_company.fit_transform(df_ml['Company Names'])
        df_ml['Fuel_Encoded'] = le_fuel.fit_transform(df_ml['Fuel Types'])
        
        # Features untuk training
        features = [
            'HorsePower', 'Torque', 'CC/Battery Capacity', 
            'Performance(0 - 100 )KM/H', 'Total Speed',
            'Company_Encoded', 'Fuel_Encoded'
        ]
        
        X = df_ml[features]
        y = df_ml['Price_Category']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return {
            'model': model,
            'scaler': scaler,
            'le_company': le_company,
            'le_fuel': le_fuel,
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'categories': list(model.classes_),
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

def predict_price_category(model_data, features):
    """Prediksi kategori harga berdasarkan input features"""
    try:
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Scale input features
        features_scaled = scaler.transform([features])
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Create probability dictionary
        prob_dict = {
            model_data['categories'][i]: round(prob * 100, 2)
            for i, prob in enumerate(probabilities)
        }
        
        return prediction, prob_dict
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None

def display_search_results(df, search_input):
    """Display search results with ML classification"""
    # Cari mobil yang match
    mask = (df['Company Names'].str.upper().str.contains(search_input.upper(), na=False) | 
            df['Cars Names'].str.upper().str.contains(search_input.upper(), na=False))
    
    matching_cars = df[mask]
    
    if len(matching_cars) == 0:
        st.warning(f"❌ Tidak ditemukan mobil dengan nama '{search_input}'")
        st.info("💡 Coba cari dengan nama merk: BMW, Toyota, Honda, Mercedes, Audi, dll.")
        
        # Tampilkan beberapa merk yang tersedia
        available_brands = df['Company Names'].unique()[:10]
        st.write("**Merk yang tersedia:**", ", ".join(available_brands))
        return
    
    # Tampilkan hasil
    st.success(f"✅ Ditemukan {len(matching_cars)} mobil untuk pencarian '{search_input}'")
    
    # Urutkan berdasarkan harga (termahal dulu)
    matching_cars = matching_cars.sort_values('Cars Prices', ascending=False)
    
    for idx, car in matching_cars.iterrows():
        with st.container():
            st.markdown('<div class="car-card">', unsafe_allow_html=True)
            
            # Header dengan kategori
            st.markdown('<div class="car-header">', unsafe_allow_html=True)
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f'<div class="car-title">{car["Company Names"]} {car["Cars Names"]}</div>', unsafe_allow_html=True)
                
                # Tampilkan kategori harga
                price_category = classify_price_range(car["Cars Prices"])
                category_class = get_category_class(price_category)
                st.markdown(f'<div class="category-badge {category_class}">{price_category}</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="price-section">', unsafe_allow_html=True)
                st.markdown('<div class="price-label">HARGA</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="price-tag">${car["Cars Prices"]:,.2f}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Spesifikasi dalam grid yang rapi
            st.markdown('<div class="specs-grid">', unsafe_allow_html=True)
            
            # Spesifikasi Utama
            st.markdown('<div class="spec-section">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📊 Spesifikasi</div>', unsafe_allow_html=True)
            st.markdown('<div class="spec-list">', unsafe_allow_html=True)
            
            st.markdown('<div class="spec-item">', unsafe_allow_html=True)
            st.markdown('<div class="spec-label">⚡ Tenaga</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="spec-value">{car["HorsePower"]:.0f} HP</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="spec-item">', unsafe_allow_html=True)
            st.markdown('<div class="spec-label">🔧 Torsi</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="spec-value">{car["Torque"]:.0f} Nm</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="spec-item">', unsafe_allow_html=True)
            st.markdown('<div class="spec-label">🏎️ 0-100 km/h</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="spec-value">{car["Performance(0 - 100 )KM/H"]:.1f}s</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Detail Teknis
            st.markdown('<div class="spec-section">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🔧 Teknis</div>', unsafe_allow_html=True)
            st.markdown('<div class="spec-list">', unsafe_allow_html=True)
            
            st.markdown('<div class="spec-item">', unsafe_allow_html=True)
            st.markdown('<div class="spec-label">🚀 Mesin</div>', unsafe_allow_html=True)
            engine_value = str(car["Engines"]) if not pd.isna(car["Engines"]) else "Unknown"
            st.markdown(f'<div class="spec-value">{engine_value}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="spec-item">', unsafe_allow_html=True)
            st.markdown('<div class="spec-label">⛽ Kapasitas</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="spec-value">{car["CC/Battery Capacity"]:.0f} CC</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="spec-item">', unsafe_allow_html=True)
            st.markdown('<div class="spec-label">📈 Kecepatan</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="spec-value">{car["Total Speed"]} km/h</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Informasi Lain
            st.markdown('<div class="spec-section">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">ℹ️ Informasi</div>', unsafe_allow_html=True)
            st.markdown('<div class="spec-list">', unsafe_allow_html=True)
            
            st.markdown('<div class="spec-item">', unsafe_allow_html=True)
            st.markdown('<div class="spec-label">⛽ Bahan Bakar</div>', unsafe_allow_html=True)
            fuel_value = str(car["Fuel Types"]) if not pd.isna(car["Fuel Types"]) else "Unknown"
            st.markdown(f'<div class="spec-value">{fuel_value}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="spec-item">', unsafe_allow_html=True)
            st.markdown('<div class="spec-label">💺 Kursi</div>', unsafe_allow_html=True)
            seats_value = str(car["Seats"]) if not pd.isna(car["Seats"]) else "Unknown"
            st.markdown(f'<div class="spec-value">{seats_value}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="spec-item">', unsafe_allow_html=True)
            st.markdown('<div class="spec-label">🏭 Merk</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="spec-value">{car["Company Names"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

def ml_prediction_section(model_data):
    """Section untuk prediksi menggunakan ML"""
    with st.container():
        st.markdown('<div class="ml-section">', unsafe_allow_html=True)
        
        st.markdown('<div class="ml-header">🤖 Prediksi Kategori Harga</div>', unsafe_allow_html=True)
        
        st.write("Masukkan spesifikasi mobil untuk memprediksi kategori harganya:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            horsepower = st.number_input("HorsePower (HP)", min_value=50, max_value=2000, value=150, step=10)
            torque = st.number_input("Torque (Nm)", min_value=100, max_value=2000, value=200, step=10)
            cc_capacity = st.number_input("Kapasitas Mesin (CC)", min_value=500, max_value=8000, value=2000, step=100)
        
        with col2:
            performance = st.number_input("0-100 km/h (detik)", min_value=2.0, max_value=20.0, value=8.0, step=0.1)
            max_speed = st.number_input("Kecepatan Maksimal (km/h)", min_value=100, max_value=400, value=200, step=10)
            
            # Select company
            companies = model_data['le_company'].classes_
            selected_company = st.selectbox("Merk Mobil", companies)
            company_encoded = model_data['le_company'].transform([selected_company])[0]
            
            # Select fuel type
            fuel_types = model_data['le_fuel'].classes_
            selected_fuel = st.selectbox("Jenis Bahan Bakar", fuel_types)
            fuel_encoded = model_data['le_fuel'].transform([selected_fuel])[0]
        
        # Predict button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button("🎯 Prediksi Sekarang", key="predict_btn", use_container_width=True)
        
        if predict_button:
            with st.spinner("Memprediksi..."):
                # Prepare features
                features = [
                    horsepower, torque, cc_capacity, 
                    performance, max_speed,
                    company_encoded, fuel_encoded
                ]
                
                # Make prediction
                prediction, probabilities = predict_price_category(model_data, features)
                
                if prediction and probabilities:
                    # Display prediction result
                    category_class = get_category_class(prediction)
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <div style="text-align: center; margin-bottom: 1rem;">
                            <h3 style="color: #4A5568; margin-bottom: 0.5rem;">Hasil Prediksi</h3>
                            <div class="category-badge {category_class}" style="font-size: 1.5rem; padding: 0.8rem 2rem;">
                                {prediction}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display probabilities
                    st.subheader("📊 Probabilitas Kategori")
                    
                    # Create bar chart for probabilities
                    prob_df = pd.DataFrame({
                        'Kategori': list(probabilities.keys()),
                        'Probabilitas (%)': list(probabilities.values())
                    }).sort_values('Probabilitas (%)', ascending=False)
                    
                    # Bar chart
                    fig = go.Figure(data=[
                        go.Bar(
                            x=prob_df['Kategori'],
                            y=prob_df['Probabilitas (%)'],
                            marker_color=['#38B2AC', '#4299E1', '#9F7AEA', '#ED8936'],
                            text=prob_df['Probabilitas (%)'].apply(lambda x: f'{x:.1f}%'),
                            textposition='auto',
                        )
                    ])
                    
                    fig.update_layout(
                        title="Probabilitas Prediksi per Kategori",
                        xaxis_title="Kategori Harga",
                        yaxis_title="Probabilitas (%)",
                        showlegend=False,
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display model accuracy
                    st.info(f"**Akurasi Model:** {model_data['accuracy']*100:.1f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)

def ml_analysis_section(df, model_data):
    """Section untuk analisis ML"""
    with st.container():
        st.markdown('<div class="ml-section">', unsafe_allow_html=True)
        
        st.markdown('<div class="ml-header">📈 Analisis Machine Learning</div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["Distribusi", "Feature Importance", "Confusion Matrix", "Model Info"])
        
        with tab1:
            # Distribution of price categories
            df['Price_Category'] = df['Cars Prices'].apply(classify_price_range)
            category_counts = df['Price_Category'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=category_counts.index,
                    values=category_counts.values,
                    hole=.3,
                    marker_colors=['#38B2AC', '#4299E1', '#9F7AEA', '#ED8936']
                )])
                
                fig.update_layout(
                    title="Distribusi Kategori Harga",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Statistics
                st.subheader("📊 Statistik Kategori")
                
                for category, count in category_counts.items():
                    percentage = (count / len(df)) * 100
                    st.metric(
                        label=category,
                        value=f"{count} mobil",
                        delta=f"{percentage:.1f}% dari total"
                    )
        
        with tab2:
            # Feature Importance
            st.subheader("🎯 Feature Importance")
            
            fig = go.Figure(data=[
                go.Bar(
                    x=model_data['feature_importance']['Importance'],
                    y=model_data['feature_importance']['Feature'],
                    orientation='h',
                    marker_color='#667eea'
                )
            ])
            
            fig.update_layout(
                title="Pengaruh Fitur terhadap Prediksi",
                xaxis_title="Importance Score",
                yaxis_title="Fitur",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **Interpretasi:**
            - Semakin tinggi score, semakin penting fitur tersebut dalam menentukan kategori harga
            - HorsePower dan Torque biasanya menjadi penentu utama
            """)
        
        with tab3:
            # Confusion Matrix
            st.subheader("📋 Confusion Matrix")
            
            # Generate predictions
            X_test_scaled = model_data['scaler'].transform(model_data['X_test'])
            y_pred = model_data['model'].predict(X_test_scaled)
            
            # Create confusion matrix
            cm = confusion_matrix(model_data['y_test'], y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=model_data['categories'],
                columns=model_data['categories']
            )
            
            # Plot heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            
            st.pyplot(fig)
            
            # Classification report
            st.subheader("Classification Report")
            report = classification_report(model_data['y_test'], y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format({
                'precision': '{:.2f}',
                'recall': '{:.2f}',
                'f1-score': '{:.2f}',
                'support': '{:.0f}'
            }))
        
        with tab4:
            # Model Information
            st.subheader("🤖 Informasi Model")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Algoritma", "Random Forest Classifier")
                st.metric("Jumlah Estimator", "200 trees")
                st.metric("Max Depth", "12")
                
            with col2:
                st.metric("Akurasi", f"{model_data['accuracy']*100:.2f}%")
                st.metric("Training Data", f"{len(model_data['X_train'])} sampel")
                st.metric("Test Data", f"{len(model_data['X_test'])} sampel")
            
            st.subheader("📚 Tentang Model")
            st.write("""
            **Random Forest Classifier** adalah algoritma ensemble learning yang:
            - Membangun banyak decision tree selama training
            - Menggunakan majority voting untuk prediksi
            - Tahan terhadap overfitting
            - Dapat menangani data non-linear dengan baik
            
            **Fitur yang digunakan:**
            1. HorsePower (Tenaga)
            2. Torque (Torsi)
            3. Kapasitas Mesin (CC)
            4. Akselerasi 0-100 km/h
            5. Kecepatan Maksimal
            6. Merk Mobil (encoded)
            7. Jenis Bahan Bakar (encoded)
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Header dengan design yang lebih clean
    st.markdown('<h1 class="main-header">🤖 MobilAI Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sistem Klasifikasi Harga Mobil dengan Machine Learning</p>', unsafe_allow_html=True)
    
    # Load data tanpa menampilkan pesan loading
    df = load_data()
    
    if df is None:
        st.error("""
        ❌ Tidak dapat memuat data. Pastikan:
        1. File 'Cars Datasets 2025.csv' ada di folder yang sama
        2. Format file CSV benar
        3. Kolom yang diperlukan tersedia
        """)
        return
    
    # Train ML model
    with st.spinner("Melatih model Machine Learning..."):
        model_data = train_ml_model(df)
    
    # Tabs untuk navigasi
    tab1, tab2, tab3 = st.tabs(["🔍 Pencarian Mobil", "🤖 Prediksi ML", "📊 Analisis Data"])
    
    with tab1:
        # Search section yang lebih rapi
        with st.container():
            st.markdown('<div class="search-container">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown('<div class="search-title">🔍 Cari mobil</div>', unsafe_allow_html=True)
                search_input = st.text_input(
                    "Masukkan merk atau model mobil:",
                    placeholder="Contoh: BMW, Toyota, Civic, Mustang...",
                    label_visibility="collapsed",
                    key="search_input"
                )
            
            with col2:
                st.markdown("<div style='margin-top: 2.2rem;'>", unsafe_allow_html=True)
                search_clicked = st.button("🚗 Cari", use_container_width=True, key="search_btn")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("<div style='margin-top: 2.2rem;'>", unsafe_allow_html=True)
                if st.button("🔄 Reset", use_container_width=True, key="reset_btn"):
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display results
        if search_input or search_clicked:
            if search_input:
                display_search_results(df, search_input)
            else:
                st.info("💡 Silakan masukkan nama mobil yang ingin dicari")
        
        # Stats section yang lebih clean
        st.markdown("## 📊 Overview Dataset")
        st.markdown('<div class="stats-container">', unsafe_allow_html=True)
        
        # Tambahkan kategori harga ke df untuk stats
        df['Price_Category'] = df['Cars Prices'].apply(classify_price_range)
        category_counts = df['Price_Category'].value_counts()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="stats-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="stats-number">{len(df):,}</div>', unsafe_allow_html=True)
            st.markdown('<div class="stats-label">Total Mobil</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="stats-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="stats-number">{df["Company Names"].nunique()}</div>', unsafe_allow_html=True)
            st.markdown('<div class="stats-label">Jumlah Merk</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="stats-card">', unsafe_allow_html=True)
            avg_price = df["Cars Prices"].mean()
            st.markdown(f'<div class="stats-number">${avg_price:,.0f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="stats-label">Harga Rata-rata</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="stats-card">', unsafe_allow_html=True)
            if model_data:
                st.markdown(f'<div class="stats-number">{model_data["accuracy"]*100:.1f}%</div>', unsafe_allow_html=True)
                st.markdown('<div class="stats-label">Akurasi Model</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="stats-number">-</div>', unsafe_allow_html=True)
                st.markdown('<div class="stats-label">Akurasi Model</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Kategori distribution
        st.subheader("📈 Distribusi Kategori Harga")
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=category_counts.index,
                    y=category_counts.values,
                    marker_color=['#38B2AC', '#4299E1', '#9F7AEA', '#ED8936'],
                    text=category_counts.values,
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                xaxis_title="Kategori Harga",
                yaxis_title="Jumlah Mobil",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Table
            category_df = pd.DataFrame({
                'Kategori': category_counts.index,
                'Jumlah': category_counts.values,
                'Persentase': (category_counts.values / len(df) * 100).round(1)
            })
            
            st.dataframe(
                category_df.style.format({
                    'Persentase': '{:.1f}%'
                }).background_gradient(subset=['Jumlah'], cmap='Blues'),
                use_container_width=True
            )
    
    with tab2:
        # ML Prediction Section
        if model_data:
            ml_prediction_section(model_data)
        else:
            st.error("Model ML belum tersedia. Silakan periksa data Anda.")
    
    with tab3:
        # ML Analysis Section
        if model_data:
            ml_analysis_section(df, model_data)
        else:
            st.error("Model ML belum tersedia. Silakan periksa data Anda.")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🤖 Teknologi:**
        - Streamlit
        - Scikit-learn
        - Pandas
        - Plotly
        """)
    
    with col2:
        st.markdown("""
        **🎯 Fitur:**
        - Pencarian Mobil
        - Klasifikasi Harga ML
        - Prediksi Real-time
        - Analisis Data
        """)
    
    with col3:
        st.markdown("""
        **📊 Dataset:**
        - {:,} mobil
        - {} merk
        - 4 kategori harga
        """.format(len(df), df['Company Names'].nunique()))
    
    st.markdown("---")
    st.caption("© 2024 MobilAI Classifier - Sistem Klasifikasi Harga Mobil dengan Machine Learning")

if __name__ == "__main__":
    main()
