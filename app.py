
# ============================================================
# IOT ENABLED DEEP LEARNING BASED SMART IRRIGATION SYSTEM
# Complete Streamlit Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

from pathlib import Path
from io import BytesIO

# Voice libraries
try:
    from streamlit_mic_recorder import speech_to_text
    VOICE_INPUT_AVAILABLE = True
except ImportError:
    VOICE_INPUT_AVAILABLE = False

try:
    from gtts import gTTS
    VOICE_OUTPUT_AVAILABLE = True
except ImportError:
    VOICE_OUTPUT_AVAILABLE = False

# LSTM library
try:
    from tensorflow.keras.models import load_model
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Irrigation Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_DIR
    / "dataset"
    / "Edge_IoT_Predictive_Irrigation_Dataset.xlsx.xlsx"
)

MODEL_DIR = PROJECT_DIR / "model"

MODEL_PATH = MODEL_DIR / "irrigation_lstm.keras"
INPUT_SCALER_PATH = MODEL_DIR / "input_scaler.pkl"
OUTPUT_SCALER_PATH = MODEL_DIR / "output_scaler.pkl"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
        linear-gradient(
            rgba(5, 30, 20, 0.94),
            rgba(7, 40, 27, 0.96)
        ),
        url("https://images.pexels.com/photos/32370506/pexels-photo-32370506.jpeg?cs=srgb&dl=pexels-antonio-di-giacomo-64078839-32370506.jpg&fm=jpg");

        background-size: cover;
        background-attachment: fixed;
    }

    [data-testid="stSidebar"] {
        background-color: #102d22;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff;
    }

    h1, h2, h3 {
        color: #d9f99d !important;
    }

    p, label, span {
        color: #e5f5e9;
    }

    [data-testid="stMetric"] {
        background: rgba(20, 70, 48, 0.85);
        border: 1px solid #477b59;
        padding: 18px;
        border-radius: 15px;
    }

    [data-testid="stMetricLabel"] {
        color: #c5e8ce !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    .stButton button {
        background-color: #387a4a;
        color: white;
        border-radius: 10px;
        border: none;
    }

    .stButton button:hover {
        background-color: #4b9b5d;
        color: white;
    }

    .hero-box {
        background: rgba(15, 65, 43, 0.85);
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #568d60;
        margin-bottom: 20px;
    }

    .hero-title {
        font-size: 30px;
        font-weight: bold;
        color: #d9f99d;
    }

    .hero-subtitle {
        font-size: 16px;
        color: #e5f5e9;
        margin-top: 8px;
    }

    .info-box {
        background: rgba(25, 75, 50, 0.8);
        padding: 18px;
        border-radius: 12px;
        border-left: 4px solid #a3e635;
        margin-bottom: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

with st.container(border=True):

    st.caption("SMART AGRICULTURE TECHNOLOGY")

    st.title("🌿 Smart Irrigation Monitoring Dashboard")

    st.write(
        "IoT Enabled Deep Learning Based Smart Irrigation System"
    )


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    if not DATASET_PATH.exists():
        return None

    data = pd.read_excel(DATASET_PATH)

    required_columns = [
        "Timestamp",
        "Temperature_C",
        "Humidity_pct",
        "Soil_Moisture_pct",
        "Light_Intensity",
        "Pump_Status"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: {missing_columns}"
        )

    data["Timestamp"] = pd.to_datetime(
        data["Timestamp"],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "Timestamp",
            "Temperature_C",
            "Humidity_pct",
            "Soil_Moisture_pct",
            "Light_Intensity",
            "Pump_Status"
        ]
    )

    data = data.sort_values("Timestamp").reset_index(drop=True)

    return data


try:
    df = load_dataset()

except Exception as error:

    st.error(f"Dataset loading error: {error}")
    st.stop()


if df is None:

    st.error(
        "Dataset not found. Please check the dataset folder."
    )

    st.code(str(DATASET_PATH))
    st.stop()


if len(df) == 0:

    st.error("The dataset does not contain valid records.")
    st.stop()


# ============================================================
# LSTM MODEL FUNCTIONS
# ============================================================

@st.cache_resource
def load_lstm_components():

    if not LSTM_AVAILABLE:
        raise ImportError("TensorFlow is not installed.")

    model = load_model(
        MODEL_PATH,
        compile=False
    )

    input_scaler = joblib.load(
        INPUT_SCALER_PATH
    )

    output_scaler = joblib.load(
        OUTPUT_SCALER_PATH
    )

    return model, input_scaler, output_scaler


def predict_next_moisture(data):

    features = [
        "Temperature_C",
        "Humidity_pct",
        "Soil_Moisture_pct",
        "Light_Intensity",
        "Pump_Status"
    ]

    if len(data) < 48:
        return None

    if not MODEL_PATH.exists():
        return None

    if not INPUT_SCALER_PATH.exists():
        return None

    if not OUTPUT_SCALER_PATH.exists():
        return None

    model, input_scaler, output_scaler = (
        load_lstm_components()
    )

    latest_data = data[features].tail(48).values

    scaled_data = input_scaler.transform(
        latest_data
    )

    model_input = scaled_data.reshape(
        1, 48, 5
    )

    prediction = model.predict(
        model_input,
        verbose=0
    )

    predicted_value = output_scaler.inverse_transform(
        prediction.reshape(-1, 1)
    )[0][0]

    return float(predicted_value)


# ============================================================
# LATEST DATA
# ============================================================

latest = df.iloc[-1]

moisture = float(latest["Soil_Moisture_pct"])

temperature = float(latest["Temperature_C"])

humidity = float(latest["Humidity_pct"])

light = float(latest["Light_Intensity"])

pump_status = (
    "ON" if int(latest["Pump_Status"]) == 1
    else "OFF"
)

threshold = 40


# ============================================================
# LSTM PREDICTION
# ============================================================

try:

    predicted_moisture = predict_next_moisture(df)

except Exception as error:

    predicted_moisture = None

    st.warning(
        f"LSTM prediction unavailable: {error}"
    )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🌿 Smart Irrigation")

st.sidebar.caption(
    "IoT Enabled Deep Learning System"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Sensor Monitoring",
        "Historical Analysis",
        "AI Farmer Assistant",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.subheader("Project Information")

st.sidebar.write("Department: CSBS")

st.sidebar.write("Batch Number: 27BS15")

st.sidebar.write("Model: LSTM")

st.sidebar.write("Platform: ESP32 + Python")


# ============================================================
# DASHBOARD PAGE
# ============================================================

if page == "Dashboard":

    st.subheader("📊 System Overview")

    st.info(
        "The displayed readings are from the historical "
        "demonstration dataset. Live ESP32 sensor monitoring "
        "is not connected yet."
    )

    # Metric cards

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "🌱 Soil Moisture",
            f"{moisture:.1f}%"
        )

    with col2:

        st.metric(
            "🌡️ Temperature",
            f"{temperature:.1f} °C"
        )

    with col3:

        st.metric(
            "💧 Humidity",
            f"{humidity:.1f}%"
        )

    col4, col5, col6 = st.columns(3)

    with col4:

        st.metric(
            "☀️ Light Intensity",
            f"{light:.1f}"
        )

    with col5:

        st.metric(
            "⚙️ Pump Status",
            pump_status
        )

    with col6:

        st.metric(
            "💧 Water Utilised",
            "Not connected"
        )

    st.divider()

    # LSTM prediction section

    st.subheader(
        "🧠 LSTM Soil Moisture Prediction"
    )

    if predicted_moisture is not None:

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Soil Moisture",
                f"{predicted_moisture:.2f}%"
            )

        with col2:

            if predicted_moisture < threshold:

                st.warning(
                    "Irrigation may be required "
                    "based on the demonstration threshold."
                )

            else:

                st.success(
                    "Predicted moisture is above "
                    "the demonstration threshold."
                )

        st.caption(
            "Prediction horizon: Next 30-minute step. "
            "Calculated using the latest 48 historical readings."
        )

    else:

        st.warning(
            "LSTM prediction is unavailable. "
            "Check the model files and dataset."
        )

    st.divider()

    # Soil moisture graph

    st.subheader("🌱 Soil Moisture Trend")

    fig = px.line(
        df,
        x="Timestamp",
        y="Soil_Moisture_pct",
        title="Historical Soil Moisture Readings",
        markers=True
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # Environmental trends

    st.subheader("🌤️ Environmental Monitoring")

    environmental_data = df[
        [
            "Timestamp",
            "Temperature_C",
            "Humidity_pct"
        ]
    ].melt(
        id_vars="Timestamp",
        var_name="Parameter",
        value_name="Value"
    )

    fig = px.line(
        environmental_data,
        x="Timestamp",
        y="Value",
        color="Parameter",
        title="Temperature and Humidity Trends"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.subheader("📋 Latest Dataset Records")

    st.dataframe(
        df.tail(10),
        width="stretch",
        hide_index=True
    )


# ============================================================
# SENSOR MONITORING PAGE
# ============================================================

elif page == "Sensor Monitoring":

    st.subheader("📡 Sensor Monitoring")

    st.info(
        "These are historical demonstration readings. "
        "Live ESP32 hardware integration is pending."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Soil Moisture Sensor",
            f"{moisture:.1f}%"
        )

        st.metric(
            "Temperature Sensor",
            f"{temperature:.1f} °C"
        )

        st.metric(
            "Humidity Sensor",
            f"{humidity:.1f}%"
        )

    with col2:

        st.metric(
            "Light Intensity",
            f"{light:.1f}"
        )

        st.metric(
            "Pump Status",
            pump_status
        )

        st.metric(
            "Water Flow Sensor",
            "Not connected"
        )

    st.divider()

    st.subheader("Sensor Reading History")

    sensor_choice = st.selectbox(
        "Select Sensor",
        [
            "Soil_Moisture_pct",
            "Temperature_C",
            "Humidity_pct",
            "Light_Intensity",
            "Pump_Status"
        ]
    )

    fig = px.line(
        df,
        x="Timestamp",
        y=sensor_choice,
        title=f"{sensor_choice} History",
        markers=True
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.subheader("Recent Sensor Records")

    st.dataframe(
        df.tail(20),
        width="stretch",
        hide_index=True
    )


# ============================================================
# HISTORICAL ANALYSIS PAGE
# ============================================================

elif page == "Historical Analysis":

    st.subheader("📈 Historical Data Analysis")

    st.write(
        "Explore the historical sensor dataset "
        "and understand environmental changes."
    )

    parameter = st.selectbox(
        "Choose Parameter",
        [
            "Soil_Moisture_pct",
            "Temperature_C",
            "Humidity_pct",
            "Light_Intensity",
            "Pump_Status"
        ]
    )

    col1, col2, col3, col4 = st.columns(4)

    selected_data = df[parameter]

    with col1:

        st.metric(
            "Minimum",
            f"{selected_data.min():.2f}"
        )

    with col2:

        st.metric(
            "Maximum",
            f"{selected_data.max():.2f}"
        )

    with col3:

        st.metric(
            "Average",
            f"{selected_data.mean():.2f}"
        )

    with col4:

        st.metric(
            "Records",
            f"{len(df)}"
        )

    st.divider()

    st.subheader(
        f"{parameter} Trend"
    )

    fig = px.line(
        df,
        x="Timestamp",
        y=parameter,
        title=f"Historical {parameter} Analysis",
        markers=True
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        df,
        width="stretch",
        hide_index=True
    )


# ============================================================
# AI FARMER ASSISTANT PAGE
# ============================================================

elif page == "AI Farmer Assistant":

    st.subheader("🤖 Tamil-English AI Farmer Assistant")

    st.write(
        "Ask your farming questions by typing or speaking."
    )

    st.info(
        "This is a rule-based assistant that uses "
        "historical sensor readings and LSTM predictions. "
        "It does not require a Gemini API key."
    )

    # Language selection

    language = st.selectbox(
        "Choose Language",
        [
            "English",
            "தமிழ்",
            "Tamil + English"
        ],
        key="farmer_language"
    )

    if language == "தமிழ்":

        speech_language = "ta-IN"
        speech_output_language = "ta"

    elif language == "Tamil + English":

        speech_language = "ta-IN"
        speech_output_language = "ta"

    else:

        speech_language = "en-IN"
        speech_output_language = "en"

    # Chat history

    if "chat_history" not in st.session_state:

        st.session_state.chat_history = []

    # Display previous messages

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):

            st.write(message["content"])

    st.divider()

    # Voice input

    st.markdown("### 🎤 Speak to the Farmer Assistant")

    voice_text = None

    if VOICE_INPUT_AVAILABLE:

        voice_text = speech_to_text(
            language=speech_language,
            start_prompt="🎤 Start Speaking",
            stop_prompt="⏹️ Stop Recording",
            just_once=True,
            use_container_width=True,
            key=f"farmer_voice_{language}"
        )

        if voice_text:

            st.success(
                f"Voice detected: {voice_text}"
            )

    else:

        st.warning(
            "Voice input library is unavailable. "
            "You can still type your question."
        )

    # Text input

    typed_question = st.chat_input(
        "Type your farming question here..."
    )

    # Process voice or typed question

    user_question = (
        voice_text if voice_text
        else typed_question
    )

    # Avoid processing the same voice result twice

    if "last_voice_question" not in st.session_state:

        st.session_state.last_voice_question = None

    if voice_text:

        if voice_text == st.session_state.last_voice_question:

            user_question = None

        else:

            st.session_state.last_voice_question = voice_text

    # Process question

    if user_question:

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        with st.chat_message("user"):

            st.write(user_question)

        question = user_question.lower().strip()

        # Historical readings

        moisture = float(latest["Soil_Moisture_pct"])

        temperature = float(latest["Temperature_C"])

        humidity = float(latest["Humidity_pct"])

        threshold = 40

        # LSTM prediction

        try:

            assistant_prediction = predict_next_moisture(df)

        except Exception:

            assistant_prediction = None

        # ====================================================
        # ENGLISH RESPONSES
        # ====================================================

        if language == "English":

            if any(word in question for word in [
                "hello", "hi", "vanakkam"
            ]):

                reply = (
                    "Hello! Welcome to the Smart Irrigation "
                    "Farmer Assistant. How can I help you?"
                )

            elif any(word in question for word in [
                "soil", "moisture"
            ]):

                reply = (
                    f"The latest historical soil moisture "
                    f"is {moisture:.1f} percent. "
                    f"The demonstration irrigation threshold "
                    f"is {threshold} percent."
                )

            elif any(word in question for word in [
                "temperature", "heat"
            ]):

                reply = (
                    f"The latest historical temperature "
                    f"is {temperature:.1f} degrees Celsius."
                )

            elif "humidity" in question:

                reply = (
                    f"The latest historical humidity "
                    f"is {humidity:.1f} percent."
                )

            elif "pump" in question:

                reply = (
                    f"The historical dataset pump status is "
                    f"{pump_status}. Live ESP32 monitoring "
                    f"is not connected yet."
                )

            elif any(word in question for word in [
                "predict", "prediction", "future"
            ]):

                if assistant_prediction is not None:

                    reply = (
                        f"The LSTM model predicts soil moisture "
                        f"of {assistant_prediction:.2f} percent "
                        f"for the next 30-minute step."
                    )

                else:

                    reply = (
                        "LSTM prediction is currently unavailable."
                    )

            elif any(word in question for word in [
                "irrigation", "water"
            ]):

                if assistant_prediction is None:

                    reply = (
                        "Irrigation recommendation is unavailable "
                        "because the LSTM prediction is not available."
                    )

                elif assistant_prediction < threshold:

                    reply = (
                        "Irrigation may be required. Predicted "
                        "soil moisture is below the 40 percent "
                        "demonstration threshold. Please verify "
                        "the actual field condition."
                    )

                elif assistant_prediction <= 50:

                    reply = (
                        "Please monitor soil moisture. "
                        "It is approaching the demonstration threshold."
                    )

                else:

                    reply = (
                        "The predicted soil moisture is above "
                        "the demonstration threshold. Continue monitoring."
                    )

            else:

                reply = (
                    "I can help with soil moisture, temperature, "
                    "humidity, pump status, LSTM prediction, "
                    "and irrigation information."
                )

        # ====================================================
        # TAMIL RESPONSES
        # ====================================================

        elif language == "தமிழ்":

            if any(word in question for word in [
                "வணக்கம்", "ஹலோ"
            ]):

                reply = (
                    "வணக்கம்! ஸ்மார்ட் நீர்ப்பாசன உதவியாளருக்கு "
                    "உங்களை வரவேற்கிறேன். என்ன உதவி வேண்டும்?"
                )

            elif any(word in question for word in [
                "மண்", "ஈரப்பதம்"
            ]):

                reply = (
                    f"கடைசியாக பதிவான மண்ணின் ஈரப்பதம் "
                    f"{moisture:.1f} சதவீதம். "
                    f"செயல்முறை விளக்கத்திற்கான நீர்ப்பாசன "
                    f"அளவுகோல் {threshold} சதவீதம்."
                )

            elif any(word in question for word in [
                "வெப்பநிலை", "வெப்பம்"
            ]):

                reply = (
                    f"கடைசியாக பதிவான வெப்பநிலை "
                    f"{temperature:.1f} டிகிரி செல்சியஸ்."
                )

            elif "ஈரப்பதம்" in question:

                reply = (
                    f"காற்றின் கடைசி பதிவு செய்யப்பட்ட ஈரப்பதம் "
                    f"{humidity:.1f} சதவீதம்."
                )

            elif "பம்ப்" in question:

                reply = (
                    f"தரவுத்தொகுப்பில் பம்பின் நிலை {pump_status}. "
                    f"நேரடி ESP32 இணைப்பு இன்னும் செய்யப்படவில்லை."
                )

            elif any(word in question for word in [
                "கணிப்பு", "எதிர்காலம்"
            ]):

                if assistant_prediction is not None:

                    reply = (
                        f"LSTM மாதிரி அடுத்த 30 நிமிட படிக்கான "
                        f"மண் ஈரப்பதத்தை {assistant_prediction:.2f} "
                        f"சதவீதமாக கணிக்கிறது."
                    )

                else:

                    reply = (
                        "LSTM கணிப்பு தற்போது கிடைக்கவில்லை."
                    )

            elif any(word in question for word in [
                "நீர்ப்பாசனம்", "தண்ணீர்"
            ]):

                if assistant_prediction is not None:

                    if assistant_prediction < threshold:

                        reply = (
                            "மண்ணின் ஈரப்பதம் குறைவாக இருக்கலாம். "
                            "நீர்ப்பாசனம் தேவைப்படலாம். "
                            "நிலத்தின் உண்மையான நிலையை சரிபார்க்கவும்."
                        )

                    else:

                        reply = (
                            "கணிக்கப்பட்ட மண்ணின் ஈரப்பதம் "
                            "அளவுகோலை விட அதிகமாக உள்ளது. "
                            "தொடர்ந்து கண்காணிக்கவும்."
                        )

                else:

                    reply = (
                        "நீர்ப்பாசன பரிந்துரை தற்போது கிடைக்கவில்லை."
                    )

            else:

                reply = (
                    "மண் ஈரப்பதம், வெப்பநிலை, காற்றின் ஈரப்பதம், "
                    "பம்ப் நிலை, LSTM கணிப்பு மற்றும் "
                    "நீர்ப்பாசனம் பற்றி கேட்கலாம்."
                )

        # ====================================================
        # TAMIL + ENGLISH RESPONSES
        # ====================================================

        else:

            if any(word in question for word in [
                "hello", "hi", "vanakkam"
            ]):

                reply = (
                    "Vanakkam! Welcome to Smart Irrigation Assistant. "
                    "Ungalukku enna help venum?"
                )

            elif any(word in question for word in [
                "soil", "moisture", "mann", "eerappatham"
            ]):

                reply = (
                    f"Latest historical soil moisture "
                    f"{moisture:.1f} percent. "
                    f"Demonstration threshold {threshold} percent. "
                    "Soil moisture-ai monitor pannunga."
                )

            elif any(word in question for word in [
                "temperature", "heat", "veppam"
            ]):

                reply = (
                    f"Latest historical temperature "
                    f"{temperature:.1f} degree Celsius. "
                    "Temperature-ai monitor pannunga."
                )

            elif any(word in question for word in [
                "humidity", "kaatru"
            ]):

                reply = (
                    f"Latest historical humidity "
                    f"{humidity:.1f} percent."
                )

            elif "pump" in question:

                reply = (
                    f"Historical dataset pump status {pump_status}. "
                    "Live ESP32 connection innum complete aagala."
                )

            elif any(word in question for word in [
                "predict", "prediction", "future", "kanippu"
            ]):

                if assistant_prediction is not None:

                    reply = (
                        f"LSTM model next 30-minute step soil moisture "
                        f"{assistant_prediction:.2f} percent-nu predict "
                        "pannudhu."
                    )

                else:

                    reply = (
                        "LSTM prediction ippo available illa."
                    )

            elif any(word in question for word in [
                "irrigation", "water", "thanneer"
            ]):

                if assistant_prediction is not None:

                    if assistant_prediction < threshold:

                        reply = (
                            "Irrigation thevai padalam. Predicted "
                            "soil moisture 40 percent threshold-ku "
                            "keela irukku. Actual field condition "
                            "check pannunga."
                        )

                    else:

                        reply = (
                            "Predicted soil moisture threshold-ku "
                            "mela irukku. Regular-ah monitor pannunga."
                        )

                else:

                    reply = (
                        "Irrigation recommendation available illa."
                    )

            else:

                reply = (
                    "Soil moisture, temperature, humidity, pump status, "
                    "LSTM prediction pathi kelunga. "
                    "Ungalukku help panren!"
                )

        # ====================================================
        # DISPLAY ASSISTANT RESPONSE
        # ====================================================

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": reply
            }
        )

        with st.chat_message("assistant"):

            st.write(reply)

            # Voice output

            if VOICE_OUTPUT_AVAILABLE:

                try:

                    audio_buffer = BytesIO()

                    tts = gTTS(
                        text=reply,
                        lang=speech_output_language
                    )

                    tts.write_to_fp(audio_buffer)

                    audio_buffer.seek(0)

                    st.audio(
                        audio_buffer,
                        format="audio/mp3"
                    )

                except Exception as error:

                    st.warning(
                        f"Voice output unavailable: {error}"
                    )

            else:

                st.caption(
                    "Install gTTS to enable spoken replies."
                )

    # Clear conversation

    st.divider()

    if st.button(
        "🗑️ Clear Chat History"
    ):

        st.session_state.chat_history = []

        st.session_state.last_voice_question = None

        st.rerun()


# ============================================================
# ABOUT PROJECT PAGE
# ============================================================

elif page == "About Project":

    st.subheader("ℹ️ About Our Project")

    st.markdown(
        """
        <div class="info-box">
        <h3>🌿 Project Title</h3>
        IoT Enabled Deep Learning Based Smart Irrigation System
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("🎯 Project Objective")

    st.write(
        "To develop an intelligent irrigation system that uses "
        "IoT sensor data and an LSTM deep learning model to "
        "predict future soil moisture and support irrigation decisions."
    )

    st.divider()

    st.subheader("⚙️ Hardware Components")

    components = [
        "ESP32 Microcontroller",
        "Capacitive Soil Moisture Sensor",
        "2-Channel Relay Module",
        "12V Water Pump",
        "12V Adapter",
        "Water Flow Sensor",
        "PCB and Jumper Wires"
    ]

    for component in components:

        st.write(f"• {component}")

    st.divider()

    st.subheader("🧠 Software Technologies")

    technologies = [
        "Python",
        "Streamlit",
        "TensorFlow",
        "LSTM Deep Learning",
        "Pandas",
        "NumPy",
        "Plotly",
        "gTTS",
        "Streamlit Mic Recorder"
    ]

    for technology in technologies:

        st.write(f"• {technology}")

    st.divider()

    st.subheader("🔄 System Workflow")

    st.markdown(
        """
        1. Sensor collects soil and environmental readings.

        2. ESP32 sends sensor data to the Python backend.

        3. Historical sensor readings are processed.

        4. LSTM predicts soil moisture for the next 30-minute step.

        5. Irrigation decision support is generated.

        6. Relay and pump control can be integrated.

        7. Water flow sensor can verify water movement.

        8. Streamlit dashboard displays readings and predictions.

        9. Tamil-English Farmer Assistant provides information.
        """
    )

    st.divider()

    st.subheader("👩‍🎓 Project Team")

    st.write("**Department:** Computer Science and Business Systems")

    st.write("**Batch Number:** 27BS15")

    st.write("**Team Members:**")

    st.write("1. Awitha M")

    st.write("2. Birundavathy R")

    st.write("3. Kirubavathy E")

    st.divider()

    st.subheader("🚧 Development Status")

    st.write(
        "Dashboard and historical dataset analysis are available."
    )

    st.write(
        "LSTM prediction is integrated into the dashboard."
    )

    st.write(
        "Rule-based Tamil-English Farmer Assistant is included."
    )

    st.write(
        "Live ESP32 sensor integration and automatic pump control "
        "are pending."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#c5e8ce; padding:15px;">

    🌿 IoT Enabled Deep Learning Based Smart Irrigation System

    <br>

    Department of Computer Science and Business Systems

    <br>

    Smart Agriculture Technology | Batch 27BS15

    </div>
    """,
    unsafe_allow_html=True
)
