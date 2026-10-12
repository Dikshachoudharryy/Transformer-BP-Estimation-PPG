import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

import time
from datetime import datetime

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
import io

import io

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch

from datetime import datetime

from model import load_model

from utils import (
    record_webcam_signal,
    preprocess_ppg,
    assess_ppg_quality,
    generate_scalogram,
    save_scalogram,
    load_scalogram_image,
    predict_bp
    
)

from xai import generate_patch_occlusion

from verification import (
    capture_face,
    face_registered,
    verify_face,
    liveness_check
)

from patient_data import (
    create_patient_profile,
    load_patient_profile,
    save_prediction,
    load_prediction_history
)



# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="AI Blood Pressure Estimation System",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 AI Blood Pressure Estimation System")

st.caption(
    "Non-invasive Blood Pressure Estimation using Webcam PPG and Deep Learning"
)

st.markdown("---")

# ----------------------------------------------------
# Load Model
# ----------------------------------------------------

@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

st.sidebar.title("⚙️ Controls")

st.sidebar.success("✅ AI Model Loaded")

st.sidebar.info("📷 Webcam Ready")

st.sidebar.markdown("---")

st.sidebar.markdown("""
### Instructions

• Sit comfortably

• Keep your face inside the camera

• Avoid sudden movement

• Ensure good lighting

• Keep your finger steady on the webcam

• Recording takes a few seconds
""")

duration = st.sidebar.slider(
    "Recording Duration (seconds)",
    min_value=10,
    max_value=60,
    value=30
)


st.sidebar.markdown("---")

st.sidebar.subheader("🤖 Model Information")

st.sidebar.write("**Architecture:** Transformer V1")

st.sidebar.write("**Task:** BP Regression")

st.sidebar.write("**Input:** 224 × 224 Scalogram")

st.sidebar.write("**Output:** SBP + DBP")

st.sidebar.write("**XAI:** Patch Occlusion")

st.sidebar.success("🟢 System Ready")

# ============================================================
# PATIENT IDENTITY VERIFICATION
# ============================================================

st.markdown("---")

st.subheader("🔐 Patient Identity Verification")


# ------------------------------------------------------------
# Session State
# ------------------------------------------------------------

if "face_verified" not in st.session_state:
    st.session_state.face_verified = False

if "liveness_verified" not in st.session_state:
    st.session_state.liveness_verified = False

if "verified_patient_id" not in st.session_state:
    st.session_state.verified_patient_id = None


# ------------------------------------------------------------
# Patient Details
# ------------------------------------------------------------

verify_col1, verify_col2 = st.columns(2)


with verify_col1:

    st.markdown("### 👤 Patient Details")

    patient_id = st.text_input(
        "Patient ID",
        placeholder="Example: P001"
    )

    patient_name = st.text_input(
        "Patient Name",
        placeholder="Enter patient name"
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=21
    )

    gender = st.selectbox(
        "Gender",
        [
            "Female",
            "Male",
            "Other"
        ]
    )


with verify_col2:

    st.markdown("### 📷 Identity Verification")

    if patient_id:

        # ----------------------------------------------------
        # Existing patient
        # ----------------------------------------------------

        if face_registered(patient_id):

            st.success(
                "✅ Face already registered for this Patient ID."
            )

            st.info(
                "The registered face will be compared with the live patient."
            )

        # ----------------------------------------------------
        # New patient
        # ----------------------------------------------------

        else:

            st.warning(
                "⚠️ No face registered for this Patient ID."
            )

            register_face = st.button(
                "📷 Register Patient Face",
                use_container_width=True
            )

            if register_face:

                face_path, message = capture_face(
                    patient_id
                )

                if face_path:

                    st.success(
                        message
                    )

                    st.image(
                        face_path,
                        caption="Registered Patient Face",
                        use_container_width=True
                    )

                else:

                    st.error(
                        message
                    )


# ============================================================
# RESET VERIFICATION WHEN PATIENT CHANGES
# ============================================================

if (
    st.session_state.verified_patient_id is not None
    and patient_id != st.session_state.verified_patient_id
):

    st.session_state.face_verified = False
    st.session_state.liveness_verified = False
    st.session_state.verified_patient_id = None


# ============================================================
# VERIFY LIVE PATIENT
# ============================================================

if (
    patient_id
    and patient_name
    and face_registered(patient_id)
):

    st.markdown("---")

    st.subheader("🔎 Verify Patient Identity")

    # --------------------------------------------------------
    # STEP 1 — FACE VERIFICATION
    # --------------------------------------------------------

    if not st.session_state.face_verified:

        st.write(
            "Please look directly at the webcam. "
            "Your face will be compared with the registered patient."
        )

        verify_identity = st.button(
            "🔐 Verify My Identity",
            use_container_width=True
        )

        if verify_identity:

            with st.spinner(
                "🔍 Verifying patient identity..."
            ):

                verified, similarity, message = verify_face(
                    patient_id
                )

            if verified:

                st.session_state.face_verified = True

                st.session_state.verified_patient_id = patient_id

                st.success(
                    "🟢 Face Identity Verified"
                )

                st.info(
                    f"Patient: {patient_name} | "
                    f"Patient ID: {patient_id}"
                )

                st.metric(
                    "Face Similarity",
                    f"{similarity * 100:.1f}%"
                )

                st.rerun()

            else:

                st.session_state.face_verified = False

                st.session_state.liveness_verified = False

                st.session_state.verified_patient_id = None

                st.error(
                    "🔴 Identity Verification Failed"
                )

                st.warning(
                    message
                )

                if similarity > 0:

                    st.metric(
                        "Face Similarity",
                        f"{similarity * 100:.1f}%"
                    )

    # --------------------------------------------------------
    # STEP 2 — LIVENESS VERIFICATION
    # --------------------------------------------------------

    if (
        st.session_state.face_verified
        and
        st.session_state.verified_patient_id == patient_id
        and
        not st.session_state.liveness_verified
    ):

        st.markdown("---")

        st.subheader(
            "👁️ Step 2 — Liveness Verification"
        )

        st.info(
            "Please look at the camera and slowly move "
            "your head LEFT or RIGHT when prompted."
        )

        check_liveness = st.button(
            "👁️ Start Liveness Check",
            use_container_width=True
        )

        if check_liveness:

            with st.spinner(
                "👁️ Checking whether the patient is live..."
            ):

                live, live_message = liveness_check()

            if live:

                st.session_state.liveness_verified = True

                create_patient_profile(
                    patient_id=patient_id,
                    name=patient_name,
                    age=age,
                    gender=gender
                )

                st.success(
                    "🟢 Liveness Verified Successfully"
                )

                st.rerun()

            else:

                st.session_state.liveness_verified = False

                st.error(
                    "🔴 Liveness Verification Failed"
                )

                st.warning(
                    live_message
                )

# ============================================================
# PATIENT SESSION STATUS
# ============================================================

start = False

st.markdown("---")

if (
    st.session_state.face_verified
    and
    st.session_state.liveness_verified
    and
    st.session_state.verified_patient_id == patient_id
):


    st.success(
        "🟢 Identity verified. "
        "Blood pressure measurement is unlocked."
    )

    lock_session = st.button(
        "🔒 Lock Patient Session",
        use_container_width=True
    )

    if lock_session:

        st.session_state.face_verified = False

        st.session_state.liveness_verified = False

        st.session_state.verified_patient_id = None

        st.rerun()

    # ============================================================
    # PATIENT DASHBOARD
    # ============================================================

    st.markdown("---")

    st.subheader("👤 Patient Dashboard")

    profile = load_patient_profile(
        patient_id
    )

    if profile:

        dashboard_col1, dashboard_col2, dashboard_col3, dashboard_col4 = st.columns(4)

        with dashboard_col1:
            st.metric(
                "👤 Patient",
                profile["name"]
            )

        with dashboard_col2:
            st.metric(
                "🆔 Patient ID",
                profile["patient_id"]
            )

        with dashboard_col3:
            st.metric(
                "🎂 Age",
                profile["age"]
            )

        with dashboard_col4:
            st.metric(
                "⚧ Gender",
                profile["gender"]
            )

        # ----------------------------------------
    # LATEST BLOOD PRESSURE
    # ============================================================

    history = load_prediction_history(
        patient_id
    )

    if not history.empty:

        latest = history.iloc[-1]

        st.markdown("---")

        st.subheader("🩸 Latest Blood Pressure")

        bp_col1, bp_col2, bp_col3 = st.columns(3)

        with bp_col1:

            st.metric(
                "Systolic BP",
                f"{latest['SBP']:.1f} mmHg"
            )

        with bp_col2:

            st.metric(
                "Diastolic BP",
                f"{latest['DBP']:.1f} mmHg"
            )

        with bp_col3:

            st.metric(
                "Last Measurement",
                str(latest["Date"])
            )

    # ============================================================
    # BLOOD PRESSURE TREND
    # ============================================================

    if len(history) >= 2:

        st.markdown("---")

        st.subheader("📊 Blood Pressure Trend")

        trend = history.copy()

        trend["Date"] = pd.to_datetime(
            trend["Date"]
        )

        trend = trend.set_index(
            "Date"
        )

        st.line_chart(
            trend[
                ["SBP", "DBP"]
            ]
        )

    else:

        st.info(
            "At least two measurements are required "
            "to display a BP trend."
        )

    # ============================================================
    # AFTER-CARE RECOMMENDATION
    # ============================================================

    st.markdown("---")

    st.subheader("💡 After-Care Guidance")

    if not history.empty:

        latest_sbp = float(
            latest["SBP"]
        )

        latest_dbp = float(
            latest["DBP"]
        )

        if latest_sbp < 90 or latest_dbp < 60:

            st.warning(
                """
                Your estimated blood pressure is in a low range.

                • Rest and avoid sudden standing.
                • Maintain adequate hydration.
                • If you experience dizziness, fainting, confusion,
                or persistent symptoms, seek medical attention.
                • Confirm the reading with a clinically validated
                blood-pressure device.
                """
            )

        elif latest_sbp < 120 and latest_dbp < 80:

            st.success(
                """
                Your estimated blood pressure is in the normal range.

                • Continue a balanced lifestyle.
                • Stay physically active.
                • Maintain healthy dietary habits.
                • Continue periodic BP monitoring.
                """
            )

        elif latest_sbp < 130 and latest_dbp < 80:

            st.info(
                """
                Your estimated reading is in an elevated range.

                • Maintain a balanced diet.
                • Limit excessive sodium intake.
                • Stay physically active.
                • Continue monitoring your BP.
                • Confirm unusual readings using a validated BP device.
                """
            )

        elif latest_sbp < 140 or latest_dbp < 90:

            st.warning(
                """
                Your estimated reading falls in a higher range.

                • Monitor your blood pressure regularly.
                • Maintain a heart-healthy lifestyle.
                • Consider discussing repeated elevated readings
                with a healthcare professional.
                • Confirm the result using a validated BP monitor.
                """
            )

        else:

            st.error(
                """
                Your estimated reading is in a high range.

                • Do not rely on this AI estimate alone.
                • Repeat the measurement using a clinically validated
                blood-pressure device.
                • If elevated readings persist, consult a healthcare
                professional.
                • Seek urgent medical care if severe symptoms occur.
                """
            )

    else:

        st.info(
            "After-care guidance will appear after your first measurement."
        )

    # ========================================================
    # START BP MEASUREMENT
    # ========================================================

    st.markdown("---")

    st.subheader("🩸 New Blood Pressure Measurement")

    st.info(
        "Identity and liveness have been verified. "
        "You can now start a new blood pressure measurement."
    )

    start = st.button(
        "▶ Start Blood Pressure Measurement",
        use_container_width=True
    )
       

else:

    start = False

    st.warning(
        "🔒 Blood pressure measurement is locked "
        "until patient identity is verified."
    )



st.markdown("---")

os.makedirs("temp", exist_ok=True)

# ----------------------------------------------------
# Start Measurement
# ----------------------------------------------------

if start:

    if patient_name == "" or patient_id == "":
        st.error("Please enter Patient ID and Patient Name.")
        st.stop()

    progress = st.progress(0)

    progress.progress(20)

    signal = record_webcam_signal(duration)

    progress.progress(35)

    # ============================================================
    # PPG SIGNAL QUALITY CHECK
    # ============================================================

    quality_score, quality_label, quality_details = assess_ppg_quality(
        signal,
        fs=30
    )

    st.markdown("---")
    st.subheader("🔍 PPG Signal Quality")

    quality_col1, quality_col2, quality_col3 = st.columns(3)

    with quality_col1:
        st.metric(
            "Signal Quality",
            quality_label
        )

    with quality_col2:
        st.metric(
            "Quality Score",
            f"{quality_score}/100"
        )

    with quality_col3:
        heart_rate = quality_details.get("heart_rate", 0)

        if heart_rate > 0:
            st.metric(
                "Estimated Pulse",
                f"{heart_rate:.1f} BPM"
            )
        else:
            st.metric(
                "Estimated Pulse",
                "Not detected"
            )

    # ------------------------------------------------------------
    # Quality message
    # ------------------------------------------------------------

    if quality_score < 50:

        st.error(
            "🔴 Poor PPG Signal Quality"
        )

        st.warning(
            """
            The webcam signal is not reliable enough for BP estimation.

            Please:
            • Place your finger steadily over the camera.
            • Cover the camera lens properly.
            • Avoid moving your finger.
            • Use stable lighting.
            • Try the measurement again.
            """
        )

        st.stop()

    elif quality_score < 65:

        st.warning(
            "🟡 Fair PPG Signal Quality — measurement may be less reliable."
        )

    else:

        st.success(
            f"🟢 {quality_label} PPG Signal — suitable for processing."
        )

    # ------------------------------------------------------------
    # Continue processing
    # ------------------------------------------------------------

    progress.progress(45)

    processed = preprocess_ppg(signal)

    
    # ----------------------------------------
    # Raw Signal
    # ----------------------------------------

    st.subheader("📈 Raw PPG Signal")

    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(signal, color="red")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Intensity")
    ax.grid(True)

    st.pyplot(fig)

    # ----------------------------------------
    # Process Signal
    # ----------------------------------------

    processed = preprocess_ppg(signal)

    st.subheader("📉 Processed PPG Signal")

    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(processed, color="blue")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Normalized Amplitude")
    ax.grid(True)

    st.pyplot(fig)

    # ----------------------------------------
    # Scalogram
    # ----------------------------------------

    scalogram = generate_scalogram(processed)

    image_path = save_scalogram(
        scalogram,
        "temp/live_scalogram.png"
    )
    progress.progress(85)


    progress.progress(100)

    st.subheader("🌊 Generated Scalogram")

    st.image(
        image_path,
        use_container_width=True
    )

    st.download_button(
        "📥 Download Scalogram",
        open(image_path, "rb"),
        file_name="scalogram.png"
    )   
    # ----------------------------------------
    # Prediction
    # ----------------------------------------

    image = load_scalogram_image(image_path)

    sbp, dbp = predict_bp(
        model,
        image
    )

    # ============================================================
    # TRANSFORMER V1 — PATCH OCCLUSION XAI
    # ============================================================

    xai_result = generate_patch_occlusion(
        model,
        image_path
    )

    sbp_attribution = xai_result["sbp_attribution"]
    dbp_attribution = xai_result["dbp_attribution"]

    st.markdown("---")
    st.subheader("🧠 Explainable AI — Patch Occlusion")

    st.write(f"Analyzed patches: {xai_result['total_patches']}")
    st.write(f"Patch size: {xai_result['patch_size']} × {xai_result['patch_size']}")

    st.markdown("### SBP Patch Attribution")
    fig_sbp, ax_sbp = plt.subplots(figsize=(6, 5))
    ax_sbp.imshow(sbp_attribution, cmap="hot", interpolation="nearest")
    ax_sbp.set_xlabel("Patch Column")
    ax_sbp.set_ylabel("Patch Row")
    st.pyplot(fig_sbp)
    plt.close(fig_sbp)

    st.markdown("### DBP Patch Attribution")
    fig_dbp, ax_dbp = plt.subplots(figsize=(6, 5))
    ax_dbp.imshow(dbp_attribution, cmap="hot", interpolation="nearest")
    ax_dbp.set_xlabel("Patch Column")
    ax_dbp.set_ylabel("Patch Row")
    st.pyplot(fig_dbp)
    plt.close(fig_dbp)

    st.info(
        "Patch Occlusion shows which regions of the scalogram have the "
        "greatest influence on the Transformer V1 blood-pressure prediction. "
        "Warmer regions indicate stronger model influence."
    )

    # ----------------------------------------
    # AI Processing Pipeline
    # ----------------------------------------

    st.markdown("---")

    st.subheader("🧠 AI Processing Pipeline")

    st.code("""
    📷 Webcam
        ↓
    ❤️ PPG Signal Extraction
        ↓
    🧹 Signal Preprocessing
        ↓
    🌊 Continuous Wavelet Transform (CWT)
        ↓
    🖼 Scalogram Generation
        ↓
    🧠 Transformer V1 Regression
        ↓
    🩸 SBP & DBP Prediction
        ↓
    💡 Explainable AI (Patch Occlusion)
        ↓
    📄 Patient Report
    """, language="text")

    import random

    confidence = None

    st.metric(
    "🤖 Model Status",
    "Prediction Successful"
    )


    st.markdown("---")


    st.subheader("🩺 Blood Pressure Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🩸 Systolic BP",
            f"{sbp:.1f} mmHg"
        )

    with col2:
        st.metric(
            "❤️ Diastolic BP",
            f"{dbp:.1f} mmHg"
        )

    with col3:
        st.metric(
            "🤖 Model Status",
            "Prediction Successful"
        )

    
    
    st.info(
        """
        The highlighted regions indicate the portions of the scalogram that had the greatest influence on the AI model's blood pressure prediction. Warmer colors represent stronger influence.
        """
        )

    # ============================================================
    # BLOOD PRESSURE CATEGORY
    # ============================================================

    st.markdown("---")

    st.subheader("📊 Blood Pressure Assessment")

    if sbp < 90 or dbp < 60:

        bp_category = "Low Blood Pressure"

        st.warning(
            "🟡 Low Blood Pressure"
        )

    elif sbp < 120 and dbp < 80:

        bp_category = "Normal"

        st.success(
            "🟢 Normal Blood Pressure"
        )

    elif sbp < 130 and dbp < 80:

        bp_category = "Elevated"

        st.info(
            "🔵 Elevated Blood Pressure"
        )

    elif sbp < 140 or dbp < 90:

        bp_category = "Hypertension Stage 1"

        st.warning(
            "🟠 Hypertension Stage 1"
        )

    else:

        bp_category = "Hypertension Stage 2"

        st.error(
            "🔴 Hypertension Stage 2"
        )

    st.markdown("---")

    st.subheader("🩺 AI Clinical Assessment")

    if sbp < 90 or dbp < 60:

        assessment = """
    Blood pressure appears lower than the normal range.

    The patient may experience dizziness or weakness.

    Repeat the measurement after resting.

    Consult a healthcare professional if symptoms persist.
    """

    elif sbp < 120 and dbp < 80:

        assessment = """
    Blood pressure appears to be within the normal range.

    No immediate abnormalities detected.

    Maintain your healthy lifestyle and continue periodic monitoring.
    """

    elif sbp < 130 and dbp < 80:

        assessment = """
    Blood pressure is slightly elevated.

    Lifestyle modification is recommended.

    Regular monitoring is advised.
    """

    else:

        assessment = """
    Blood pressure appears elevated.

    Repeat the measurement after 5–10 minutes of rest.

    Consult a healthcare professional for confirmation.
    """

    st.success(assessment)

    st.markdown("---")

    st.subheader("💡 After-care Recommendations")

    if sbp < 90 or dbp < 60:

        st.warning("""
    • Drink adequate fluids.

    • Avoid standing suddenly.

    • Eat balanced meals.

    • Seek medical advice if symptoms continue.
    """)

    elif sbp < 120 and dbp < 80:

        st.success("""
    • Continue regular exercise.

    • Maintain a balanced diet.

    • Drink enough water.

    • Sleep 7–8 hours daily.

    • Monitor BP periodically.
    """)

    elif sbp < 130 and dbp < 80:

        st.info("""
    • Reduce salt intake.

    • Exercise regularly.

    • Maintain healthy body weight.

    • Monitor BP weekly.
    """)

    else:

        st.error("""
    • Reduce sodium intake.

    • Avoid smoking and alcohol.

    • Exercise regularly.

    • Monitor BP daily.

    • Consult a physician.
    """)



    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    title_style.alignment = TA_CENTER

    heading = styles["Heading2"]

    normal = styles["BodyText"]

    story = []

    # ------------------------------------------------
    # Header
    # ------------------------------------------------

    story.append(
        Paragraph(
            "AI BLOOD PRESSURE ESTIMATION REPORT",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Webcam-based Non-invasive Blood Pressure Estimation",
            normal
        )
    )

    story.append(Spacer(1, 0.3*inch))

    # ------------------------------------------------
    # Patient Details Table
    # ------------------------------------------------

    patient_table = Table([
        ["Patient Name", patient_name],
        ["Patient ID", patient_id],
        ["Age", str(age)],
        ["Gender", gender],
        ["Date",
        datetime.now().strftime("%d-%m-%Y %H:%M")]
    ])

    patient_table.setStyle(TableStyle([

        ("GRID",(0,0),(-1,-1),1,colors.black),

        ("BACKGROUND",(0,0),(0,-1),colors.lightgrey),

        ("FONTNAME",(0,0),(-1,-1),"Helvetica"),

        ("BOTTOMPADDING",(0,0),(-1,-1),8),

    ]))

    story.append(patient_table)

    story.append(Spacer(1,0.3*inch))

    # ------------------------------------------------
    # Prediction Table
    # ------------------------------------------------

    story.append(
        Paragraph("Prediction Results", heading)
    )

    prediction_table = Table([

        ["Systolic BP (SBP)", f"{sbp:.1f} mmHg"],

        ["Diastolic BP (DBP)", f"{dbp:.1f} mmHg"],

        ["Model Status", "Prediction Successful"]

    ])

    prediction_table.setStyle(TableStyle([

        ("GRID",(0,0),(-1,-1),1,colors.black),

        ("BACKGROUND",(0,0),(0,-1),colors.lightblue),

        ("BOTTOMPADDING",(0,0),(-1,-1),8),

    ]))

    story.append(prediction_table)

    story.append(Spacer(1,0.25*inch))

    story.append(
        Paragraph(
            "AI Clinical Assessment",
            heading
        )
    )

    story.append(
        Paragraph(
            "<b>Explainable AI (Patch Occlusion)</b>",
            heading
        )
    )

    story.append(
        Paragraph(
            "Patch Occlusion analysis highlights the regions of the scalogram "
            "that had the greatest influence on the Transformer V1 prediction. "
            "Warmer colors indicate greater influence on the predicted blood pressure values.",
            normal
        )
    )

    story.append(
        Paragraph(
            assessment.replace("\n","<br/>"),
            normal
        )
    )

    story.append(Spacer(1,0.25*inch))


    story.append(
        Paragraph(
            "After-care Recommendations",
            heading
        )
    )

    if sbp < 90 or dbp < 60:

        rec = """
    • Drink sufficient fluids.<br/>
    • Avoid standing suddenly.<br/>
    • Seek medical attention if symptoms persist.
    """

    elif sbp < 120 and dbp < 80:

        rec = """
    • Continue a healthy lifestyle.<br/>
    • Exercise regularly.<br/>
    • Sleep 7–8 hours.<br/>
    • Monitor BP periodically.
    """

    elif sbp < 140:

        rec = """
    • Reduce salt intake.<br/>
    • Exercise regularly.<br/>
    • Monitor blood pressure weekly.
    """

    else:

        rec = """
    • Reduce sodium intake.<br/>
    • Consult a physician.<br/>
    • Monitor BP daily.<br/>
    • Maintain a healthy lifestyle.
    """

    story.append(
            Paragraph(
                rec,
                normal
            )
        )

    story.append(Spacer(1,0.3*inch))


    story.append(
        Paragraph(
            "<b>Disclaimer</b>",
            heading
        )
    )
    story.append(
        Paragraph(
            "This report is generated by an Artificial Intelligence system for research and educational purposes only. "
            "It should not be considered a substitute for professional medical diagnosis or treatment.",
            normal
        )
    )

    doc.build(story)

    pdf = buffer.getvalue()

    buffer.close()

    st.download_button(

        "📄 Download Patient Report",

        data=pdf,

        file_name=f"{patient_id}_BP_Report.pdf",

        mime="application/pdf",

        use_container_width=True

    )

    
    st.info(
        "⚠️ This prediction is generated by an AI model for research/demo purposes only and should not be used as a medical diagnosis."
    )

    # ----------------------------------------------------
    # Gauge Charts
    # ----------------------------------------------------

    gauge1, gauge2 = st.columns(2)

    with gauge1:

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sbp,
            title={"text": "Systolic BP"},
            gauge={
                "axis": {"range": [50, 200]},
                "bar": {"color": "red"}
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

    with gauge2:

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=dbp,
            title={"text": "Diastolic BP"},
            gauge={
                "axis": {"range": [30, 140]},
                "bar": {"color": "blue"}
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------
    # Interpretation
    # ----------------------------------------------------

    st.subheader("📋 Interpretation")

    if sbp < 90 or dbp < 60:
        st.warning("Possible Low Blood Pressure")

    elif sbp < 120 and dbp < 80:
        st.success("Blood Pressure appears to be in the normal range.")

    elif sbp < 140:
        st.info("Elevated Blood Pressure")

    else:
        st.error("High Blood Pressure")


    

    # ============================================================
    # SAVE PATIENT-SPECIFIC PREDICTION
    # ============================================================

    save_prediction(
        patient_id=patient_id,
        sbp=sbp,
        dbp=dbp
    )

    st.markdown("---")

    # ============================================================
    # PATIENT-SPECIFIC HISTORY
    # ============================================================

    st.markdown("---")

    st.subheader("📜 My Blood Pressure History")

    history = load_prediction_history(
        patient_id
    )

    if history.empty:

        st.info(
            "No previous blood pressure measurements found."
        )

    else:

        st.dataframe(
            history.tail(10),
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------------
        # BP Trend
        # --------------------------------------------------------

        st.subheader("📊 Blood Pressure Trend")

        chart_data = history.copy()

        chart_data["Date"] = pd.to_datetime(
            chart_data["Date"]
        )

        chart_data = chart_data.set_index(
            "Date"
        )

        st.line_chart(
            chart_data[
                ["SBP", "DBP"]
            ]
        )
        

        st.success("✅ Blood Pressure Measurement Completed Successfully!")

    

    # ============================================================
    # PATIENT PROFILE
    # ============================================================

    st.markdown("---")

    st.subheader("👤 Patient Profile")

    profile_col1, profile_col2 = st.columns(2)

    with profile_col1:
        st.markdown(
            f"""
            **Patient Name:** {patient_name}

            **Patient ID:** {patient_id}

            **Age:** {age} years
            """
        )

    with profile_col2:
        st.markdown(
            f"""
            **Gender:** {gender}

            **Date:** {datetime.now().strftime("%d %b %Y")}

            **Time:** {datetime.now().strftime("%I:%M %p")}
            """
        )
    st.markdown("---")

    st.caption(
        " AI Blood Pressure Estimation using Webcam PPG and Deep Learning"
    )                                                                                                                      