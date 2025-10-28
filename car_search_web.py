import streamlit as st
import pandas as pd
import numpy as np
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="Cari Harga Mobil",
    page_icon="🚗",
    layout="wide"
)

# CSS custom yang lebih clean dan modern
st.markdown("""
<style>
    /* Reset dan Base Styles */
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Header Styles */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        text-align: center;
        color: #64748b;
        margin-bottom: 3rem;
        font-size: 1.1rem;
        font-weight: 400;
        line-height: 1.6;
    }
    
    /* Search Section */
    .search-container {
        background: white;
        padding: 2.5rem;
        border-radius: 16px;
        margin: 2rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
    }
    
    .search-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Car Card Styles */
    .car-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .car-card:hover {
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        transform: translateY(-2px);
    }
    
    .car-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 2px solid #f1f5f9;
    }
    
    .car-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0;
        line-height: 1.3;
    }
    
    .price-section {
        text-align: right;
    }
    
    .price-label {
        font-size: 0.875rem;
        color: #64748b;
        font-weight: 500;
        margin-bottom: 0.25rem;
    }
    
    .price-tag {
        font-size: 1.8rem;
        font-weight: 800;
        color: #dc2626;
        line-height: 1;
    }
    
    /* Specification Grid */
    .specs-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 2rem;
    }
    
    .spec-section {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #f1f5f9;
    }
    
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 1.25rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #e2e8f0;
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
        border-bottom: 1px solid #f1f5f9;
    }
    
    .spec-label {
        color: #64748b;
        font-weight: 500;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .spec-value {
        color: #1e293b;
        font-weight: 600;
        font-size: 0.95rem;
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
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid #f1f5f9;
    }
    
    .stats-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .stats-label {
        font-size: 0.875rem;
        color: #64748b;
        font-weight: 500;
    }
    
    /* Button Styles */
    .stButton button {
        background: linear-gradient(135deg, #2563eb, #3b82f6);
        color: white;
        border: none;
        padding: 0.875rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    
    .secondary-button {
        background: #64748b !important;
    }
    
    .secondary-button:hover {
        background: #475569 !important;
    }
    
    /* Input Styles */
    .stTextInput input {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        padding: 0.875rem 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Loading Message Styles - Hidden */
    .loading-message {
        display: none !important;
    }
    
    /* Hide Streamlit Elements */
    .stAlert {
        display: none;
    }
    
    /* Utility Classes */
    .text-center { text-align: center; }
    .text-muted { color: #64748b; }
    .font-semibold { font-weight: 600; }
    .font-bold { font-weight: 700; }
    
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

def display_search_results(df, search_input):
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
            
            # Header
            st.markdown('<div class="car-header">', unsafe_allow_html=True)
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f'<div class="car-title">{car["Company Names"]} {car["Cars Names"]}</div>', unsafe_allow_html=True)
            
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
            st.markdown(f'<div class="spec-value">{car["CC/Battery Capacity"]:.0f}</div>', unsafe_allow_html=True)
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

def main():
    # Header dengan design yang lebih clean
    st.markdown('<h1 class="main-header">Cari Harga Mobil</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Temukan informasi lengkap dan harga terbaru mobil impian Anda dengan cepat dan mudah</p>', unsafe_allow_html=True)
    
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
    
    # Search section yang lebih rapi
    with st.container():
        st.markdown('<div class="search-container">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown('<div class="search-title">🔍 Cari mobil</div>', unsafe_allow_html=True)
            search_input = st.text_input(
                "Masukkan merk atau model mobil:",
                placeholder="Contoh: BMW, Toyota, Civic, Mustang...",
                label_visibility="collapsed"
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
        max_price = df["Cars Prices"].max()
        st.markdown(f'<div class="stats-number">${max_price:,.0f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stats-label">Harga Tertinggi</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Informasi tambahan
    with st.expander("ℹ️ Tentang Aplikasi"):
        st.write("""
        **Fitur Aplikasi:**
        - 🔍 Pencarian mobil berdasarkan merk dan model
        - 💰 Informasi harga dan spesifikasi lengkap
        - 📊 Statistik dataset mobil
        - 🎨 Interface yang modern dan responsif
        
        **Cara Penggunaan:**
        1. Masukkan nama merk atau model mobil di kolom pencarian
        2. Klik tombol "Cari" atau tekan Enter
        3. Lihat hasil pencarian dengan informasi lengkap
        
        **Teknologi:** Streamlit, Pandas, Python
        """)

if __name__ == "__main__":
    main()