import streamlit as st
import pandas as pd
import joblib

# Page Config
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>

/* Title */
.big-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #2563eb;
    margin-bottom: 10px;
}

/* Subtitle */
.sub-title {
    text-align: center;
    color: #94a3b8;
    font-size: 18px;
    margin-bottom: 20px;
}

/* Price Card */
.price-card {
    background: linear-gradient(135deg, #2563eb, #06b6d4);
    color: white;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    margin-top: 20px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)

# Load Model
model = joblib.load("laptop_model.pkl")

# Sidebar
st.sidebar.title("💻 Laptop Price Predictor")

st.sidebar.markdown("""
### Project Info

- 🤖 Machine Learning
- 🌲 Random Forest Regressor
- 📈 R² Score: 0.81
- 💾 Dataset: 893 Laptops

---
Built using Streamlit & Scikit-Learn
""")

# Main Title
st.markdown(
    '<div class="big-title">💻 Laptop Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Predict laptop prices using Machine Learning</div>',
    unsafe_allow_html=True
)

st.info(
    "📊 Enter laptop specifications below and click Predict Price."
)

# Layout
col1, col2 = st.columns(2)

with col1:
    brand = st.text_input("Brand", "HP")
    name = st.text_input("Laptop Name", "Victus")
    spec_rating = st.number_input("Spec Rating", value=70.0)
    processor = st.text_input("Processor", "Ryzen 5 5600H")
    cpu = st.text_input("CPU", "Hexa Core")
    ram = st.number_input("RAM (GB)", value=8)
    ram_type = st.text_input("RAM Type", "DDR4")

with col2:
    rom = st.number_input("ROM (GB)", value=512)
    rom_type = st.text_input("ROM Type", "SSD")
    gpu = st.text_input("GPU", "RTX 3050")
    display_size = st.number_input("Display Size", value=15.6)
    resolution_width = st.number_input("Resolution Width", value=1920)
    resolution_height = st.number_input("Resolution Height", value=1080)
    os = st.text_input("Operating System", "Windows 11")
    warranty = st.number_input("Warranty (Years)", value=1)

# Predict Button
if st.button("🚀 Predict Price"):

    data = pd.DataFrame({
        "brand": [brand],
        "name": [name],
        "spec_rating": [spec_rating],
        "processor": [processor],
        "CPU": [cpu],
        "Ram": [ram],
        "Ram_type": [ram_type],
        "ROM": [rom],
        "ROM_type": [rom_type],
        "GPU": [gpu],
        "display_size": [display_size],
        "resolution_width": [resolution_width],
        "resolution_height": [resolution_height],
        "OS": [os],
        "warranty": [warranty]
    })

    prediction = model.predict(data)

    st.markdown(
        f"""
        <div class="price-card">
            💰 Predicted Laptop Price<br><br>
            ₹{prediction[0]:,.0f}
        </div>
        """,
        unsafe_allow_html=True
    )

# Footer
st.markdown("---")

st.markdown(
    """
    <center>
        <h4>Developed by Aniket Sahu</h4>
        <p>B.Tech Electronics & Computer Science Engineering</p>
    </center>
    """,
    unsafe_allow_html=True
)