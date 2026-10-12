import cv2
import pywt
import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, find_peaks
from sklearn.preprocessing import MinMaxScaler

from PIL import Image
from torchvision import transforms

device = torch.device("cpu")

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor()

])

def butter_bandpass_filter(signal, fs):

    low = 0.5
    high = 8.0

    b, a = butter(
        4,
        [low/(fs/2), high/(fs/2)],
        btype="band"
    )

    return filtfilt(b, a, signal)

def preprocess_ppg(ppg, fs=30):

    filtered = butter_bandpass_filter(
        ppg,
        fs
    )

    scaler = MinMaxScaler()

    processed = scaler.fit_transform(
        filtered.reshape(-1,1)
    ).flatten()

    return processed

def assess_ppg_quality(ppg, fs=30):
    """
    Estimate the quality of a webcam-based PPG signal.

    Returns:
        quality_score : 0-100
        quality_label : Excellent / Good / Fair / Poor
        details       : dictionary of signal metrics
    """

    ppg = np.asarray(ppg, dtype=np.float64)

    # ---------------------------------------------------------
    # Basic validation
    # ---------------------------------------------------------

    if len(ppg) < fs * 10:
        return 0, "Poor", {
            "reason": "Signal is too short."
        }

    if not np.all(np.isfinite(ppg)):
        return 0, "Poor", {
            "reason": "Signal contains invalid values."
        }

    # ---------------------------------------------------------
    # Remove DC component
    # ---------------------------------------------------------

    signal = ppg - np.mean(ppg)

    # ---------------------------------------------------------
    # Band-pass filter
    # ---------------------------------------------------------

    try:
        filtered = butter_bandpass_filter(signal, fs)
    except Exception:
        return 0, "Poor", {
            "reason": "Signal filtering failed."
        }

    # ---------------------------------------------------------
    # Basic signal statistics
    # ---------------------------------------------------------

    std = float(np.std(filtered))
    amplitude = float(np.ptp(filtered))

    if std < 1e-6 or amplitude < 1e-6:
        return 0, "Poor", {
            "reason": "Very little pulsatile variation detected."
        }

    # ---------------------------------------------------------
    # Detect pulse peaks
    # ---------------------------------------------------------

    peaks, properties = find_peaks(
        filtered,
        distance=int(fs * 0.40),
        prominence=max(std * 0.20, 1e-6)
    )

    number_of_peaks = len(peaks)

    # ---------------------------------------------------------
    # Heart-rate estimate
    # ---------------------------------------------------------

    heart_rate = 0.0

    if number_of_peaks >= 2:

        intervals = np.diff(peaks) / fs

        median_interval = np.median(intervals)

        if median_interval > 0:
            heart_rate = 60.0 / median_interval

    # ---------------------------------------------------------
    # Regularity of pulse intervals
    # ---------------------------------------------------------

    regularity = 0.0

    if number_of_peaks >= 3:

        intervals = np.diff(peaks) / fs

        mean_interval = np.mean(intervals)

        if mean_interval > 0:
            cv = np.std(intervals) / mean_interval

            regularity = max(
                0.0,
                min(1.0, 1.0 - cv)
            )

    # ---------------------------------------------------------
    # Signal quality score
    # ---------------------------------------------------------

    score = 0.0

    # Signal variation
    if amplitude > 2.0:
        score += 25
    elif amplitude > 0.5:
        score += 15
    else:
        score += 5

    # Number of detected pulses
    duration_seconds = len(ppg) / fs

    expected_min_peaks = max(
        3,
        int(duration_seconds * 40 / 60)
    )

    expected_max_peaks = int(
        duration_seconds * 180 / 60
    )

    if expected_min_peaks <= number_of_peaks <= expected_max_peaks:
        score += 30
    elif number_of_peaks >= 2:
        score += 15

    # Heart rate plausibility
    if 40 <= heart_rate <= 180:
        score += 25
    elif heart_rate > 0:
        score += 10

    # Pulse regularity
    score += 20 * regularity

    score = int(round(min(100, max(0, score))))

    # ---------------------------------------------------------
    # Quality category
    # ---------------------------------------------------------

    if score >= 80:
        quality_label = "Excellent"

    elif score >= 65:
        quality_label = "Good"

    elif score >= 50:
        quality_label = "Fair"

    else:
        quality_label = "Poor"

    details = {
        "heart_rate": round(heart_rate, 1),
        "peaks_detected": number_of_peaks,
        "signal_amplitude": round(amplitude, 4),
        "signal_std": round(std, 4),
        "regularity": round(regularity, 3)
    }

    return score, quality_label, details

def record_webcam_signal(
        duration=30,
        fps=30
):

    cap = cv2.VideoCapture(0)

    red_signal = []

    

    print("Place finger on webcam...")

    total_frames = duration * fps

    while len(red_signal) < total_frames:

        ret, frame = cap.read()

        if not ret:
            continue

        red = frame[:,:,2]

        value = np.mean(red)

        red_signal.append(value)

        cv2.imshow(
            "Recording",
            frame
        )

        if cv2.waitKey(1)==27:
            break

    cap.release()

    cv2.destroyAllWindows()

    return np.array(red_signal)

def plot_signal(signal,title):

    plt.figure(figsize=(15,4))

    plt.plot(signal)

    plt.title(title)

    plt.xlabel("Samples")

    plt.ylabel("Amplitude")

    plt.grid(True)

    plt.show()


# PART 2 : Generate Scalogram

def generate_scalogram(
        signal,
        wavelet="morl",
        scales=np.arange(1,129)
):

    coef, freq = pywt.cwt(
        signal,
        scales,
        wavelet
    )

    image = np.abs(coef)

    image = image - image.min()

    image = image / (image.max() + 1e-8)

    return image

def save_scalogram(
        scalogram,
        save_path="temp/live_scalogram.png"
):

    img = (scalogram * 255).astype(np.uint8)

    img = cv2.applyColorMap(
        img,
        cv2.COLORMAP_JET
    )

    img = cv2.resize(
        img,
        (224,224)
    )

    cv2.imwrite(
        save_path,
        img
    )

    return save_path

def plot_scalogram(scalogram):

    plt.figure(figsize=(6,6))

    plt.imshow(
        scalogram,
        cmap="jet",
        aspect="auto"
    )

    plt.axis("off")

    plt.title("Generated Scalogram")

    plt.show()

def load_scalogram_image(path):

    image = Image.open(path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    return image

def predict_bp(
        model,
        image_tensor
):
    """
    Transformer V1 blood-pressure prediction.

    The Transformer outputs normalized SBP and DBP.
    This function converts them back to mmHg using
    the normalization values stored in the V1 checkpoint.
    """

    model.eval()

    # --------------------------------------------------------
    # Use the same device as the loaded model
    # --------------------------------------------------------

    model_device = next(model.parameters()).device

    image_tensor = image_tensor.to(model_device)

    # --------------------------------------------------------
    # Single forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(image_tensor)

    output = output.squeeze(0)

    # --------------------------------------------------------
    # Normalized predictions
    # --------------------------------------------------------

    sbp_normalized = output[0].item()
    dbp_normalized = output[1].item()

    # --------------------------------------------------------
    # V1 normalization values
    # --------------------------------------------------------

    SBP_MEAN = 122.304856938613
    SBP_STD = 21.051077265426

    DBP_MEAN = 52.261745783263
    DBP_STD = 11.949080997791

    # --------------------------------------------------------
    # Convert back to mmHg
    # --------------------------------------------------------

    sbp = (
        sbp_normalized * SBP_STD
        + SBP_MEAN
    )

    dbp = (
        dbp_normalized * DBP_STD
        + DBP_MEAN
    )

    return float(sbp), float(dbp)

def display_prediction(
        sbp,
        dbp
):

    print("="*50)

    print("Blood Pressure Prediction")

    print("="*50)

    print(f"SBP : {sbp:.2f} mmHg")

    print(f"DBP : {dbp:.2f} mmHg")

    print("="*50)