import os
import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = "models"
PATIENT_FACE_DIR = "patient_faces"

os.makedirs(PATIENT_FACE_DIR, exist_ok=True)


YUNET_MODEL = os.path.join(
    MODEL_DIR,
    "face_detection_yunet_2023mar.onnx"
)

SFACE_MODEL = os.path.join(
    MODEL_DIR,
    "face_recognition_sface_2021dec.onnx"
)


# ============================================================
# LOAD FACE MODELS
# ============================================================

face_detector = cv2.FaceDetectorYN.create(
    YUNET_MODEL,
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)


face_recognizer = cv2.FaceRecognizerSF.create(
    SFACE_MODEL,
    ""
)


# ============================================================
# PATIENT FACE PATH
# ============================================================

def get_patient_face_path(patient_id):

    return os.path.join(
        PATIENT_FACE_DIR,
        f"{patient_id}.jpg"
    )


# ============================================================
# CHECK REGISTERED FACE
# ============================================================

def face_registered(patient_id):

    return os.path.exists(
        get_patient_face_path(patient_id)
    )


# ============================================================
# DETECT FACE
# ============================================================

def detect_face(frame):

    height, width = frame.shape[:2]

    face_detector.setInputSize(
        (width, height)
    )

    _, faces = face_detector.detect(frame)

    if faces is None:
        return None

    if len(faces) == 0:
        return None

    # Select largest detected face
    largest_face = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    return largest_face


# ============================================================
# ALIGN FACE
# ============================================================

def align_face(frame, face):

    aligned_face = face_recognizer.alignCrop(
        frame,
        face
    )

    return aligned_face


# ============================================================
# GET FACE FEATURE
# ============================================================

def get_face_feature(frame, face):

    aligned_face = align_face(
        frame,
        face
    )

    feature = face_recognizer.feature(
        aligned_face
    )

    return feature


# ============================================================
# CAPTURE FACE FROM WEBCAM
# ============================================================

def capture_face(patient_id):

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        return None, "❌ Could not access webcam."

    captured_frame = None

    window_name = "Patient Identity Verification"

    while True:

        ret, frame = camera.read()

        if not ret:
            continue

        display_frame = frame.copy()

        face = detect_face(frame)

        if face is not None:

            x, y, w, h = face[:4].astype(int)

            cv2.rectangle(
                display_frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                "Face detected",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                display_frame,
                "No face detected",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.putText(
            display_frame,
            "SPACE = Capture | ESC = Cancel",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            window_name,
            display_frame
        )

        key = cv2.waitKey(1) & 0xFF

        # SPACE
        if key == 32:

            if face is None:

                continue

            captured_frame = frame.copy()

            break

        # ESC
        elif key == 27:

            camera.release()
            cv2.destroyAllWindows()

            return None, "Face capture cancelled."

    camera.release()
    cv2.destroyAllWindows()

    # --------------------------------------------------------
    # Save face
    # --------------------------------------------------------

    path = get_patient_face_path(
        patient_id
    )

    cv2.imwrite(
        path,
        captured_frame
    )

    return (
        path,
        "✅ Patient face registered successfully."
    )


# ============================================================
# VERIFY LIVE FACE
# ============================================================

def verify_face(patient_id):

    # ========================================================
    # SETTINGS
    # ========================================================

    # OpenCV's published SFace benchmark threshold is 0.363.
    # We intentionally use a stricter prototype threshold here.
    VERIFICATION_THRESHOLD = 0.55

    REQUIRED_GOOD_FRAMES = 4
    TOTAL_FRAMES = 6


    # ========================================================
    # REGISTERED FACE
    # ========================================================

    registered_path = get_patient_face_path(
        patient_id
    )

    if not os.path.exists(registered_path):

        return (
            False,
            0.0,
            "❌ No registered face found."
        )


    registered_image = cv2.imread(
        registered_path
    )

    if registered_image is None:

        return (
            False,
            0.0,
            "❌ Could not read registered face."
        )


    registered_face = detect_face(
        registered_image
    )

    if registered_face is None:

        return (
            False,
            0.0,
            "❌ No face detected in registered image."
        )


    # ========================================================
    # REGISTERED FACE FEATURE
    # ========================================================

    registered_feature = get_face_feature(
        registered_image,
        registered_face
    )


    # ========================================================
    # OPEN WEBCAM
    # ========================================================

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        return (
            False,
            0.0,
            "❌ Could not access webcam."
        )


    similarities = []

    good_frames = 0

    frames_checked = 0


    # ========================================================
    # LIVE VERIFICATION
    # ========================================================

    window_name = "Patient Identity Verification"

    while frames_checked < TOTAL_FRAMES:

        ret, frame = camera.read()

        if not ret:

            continue


        display_frame = frame.copy()


        live_face = detect_face(
            frame
        )


        # ----------------------------------------------------
        # NO FACE
        # ----------------------------------------------------

        if live_face is None:

            cv2.putText(
                display_frame,
                "No face detected",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.imshow(
                window_name,
                display_frame
            )

            if cv2.waitKey(1) & 0xFF == 27:

                camera.release()
                cv2.destroyAllWindows()

                return (
                    False,
                    0.0,
                    "Verification cancelled."
                )

            continue


        # ----------------------------------------------------
        # DRAW FACE BOX
        # ----------------------------------------------------

        x, y, w, h = (
            live_face[:4]
            .astype(int)
        )


        cv2.rectangle(
            display_frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # ----------------------------------------------------
        # CREATE LIVE FEATURE
        # ----------------------------------------------------

        live_feature = get_face_feature(
            frame,
            live_face
        )


        similarity = face_recognizer.match(
            registered_feature,
            live_feature,
            cv2.FaceRecognizerSF_FR_COSINE
        )


        similarity = float(
            similarity
        )


        similarities.append(
            similarity
        )


        frames_checked += 1


        if similarity >= VERIFICATION_THRESHOLD:

            good_frames += 1


        # ----------------------------------------------------
        # DISPLAY SCORE
        # ----------------------------------------------------

        cv2.putText(
            display_frame,
            f"Similarity: {similarity:.3f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        cv2.putText(
            display_frame,
            f"Frame {frames_checked}/{TOTAL_FRAMES}",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.putText(
            display_frame,
            f"Passed: {good_frames}/{REQUIRED_GOOD_FRAMES}",
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.imshow(
            window_name,
            display_frame
        )


        key = cv2.waitKey(150) & 0xFF


        if key == 27:

            camera.release()
            cv2.destroyAllWindows()

            return (
                False,
                0.0,
                "Verification cancelled."
            )


    # ========================================================
    # CLOSE CAMERA
    # ========================================================

    camera.release()

    cv2.destroyAllWindows()


    # ========================================================
    # NO VALID FRAMES
    # ========================================================

    if len(similarities) == 0:

        return (
            False,
            0.0,
            "❌ No valid face frames captured."
        )


    # ========================================================
    # CALCULATE FINAL SCORE
    # ========================================================

    similarities = np.array(
        similarities
    )


    average_similarity = float(
        np.mean(similarities)
    )


    minimum_similarity = float(
        np.min(similarities)
    )


    maximum_similarity = float(
        np.max(similarities)
    )


    # ========================================================
    # FINAL DECISION
    # ========================================================

    verified = (

        good_frames >= REQUIRED_GOOD_FRAMES

        and

        average_similarity >= VERIFICATION_THRESHOLD

        and

        minimum_similarity >= 0.45
    )


    # ========================================================
    # RESULT
    # ========================================================

    if verified:

        message = (
            "🟢 Patient identity verified successfully."
        )

    else:

        message = (
            "🔴 Identity verification failed. "
            "The live face did not sufficiently match "
            "the registered patient."
        )


    return (
        verified,
        average_similarity,
        message
    )

# ============================================================
# LIVENESS CHECK
# ============================================================

def liveness_check():

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        return False, "❌ Could not access webcam."

    # --------------------------------------------------------
    # Capture initial face position
    # --------------------------------------------------------

    initial_x = None
    movement_detected = False

    window_name = "Liveness Verification"

    frame_count = 0

    while frame_count < 80:

        ret, frame = camera.read()

        if not ret:
            continue

        display_frame = frame.copy()

        face = detect_face(frame)

        if face is None:

            cv2.putText(
                display_frame,
                "Face not detected",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.imshow(
                window_name,
                display_frame
            )

            if cv2.waitKey(30) & 0xFF == 27:

                camera.release()
                cv2.destroyAllWindows()

                return False, "Liveness check cancelled."

            continue

        # ----------------------------------------------------
        # Face coordinates
        # ----------------------------------------------------

        x, y, w, h = (
            face[:4].astype(int)
        )

        center_x = x + w // 2

        # Store initial position

        if initial_x is None:

            initial_x = center_x

        # ----------------------------------------------------
        # Calculate movement
        # ----------------------------------------------------

        movement = abs(
            center_x - initial_x
        )

        # ----------------------------------------------------
        # Draw face
        # ----------------------------------------------------

        cv2.rectangle(
            display_frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # Instructions
        # ----------------------------------------------------

        cv2.putText(
            display_frame,
            "LIVENESS CHECK",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            "Slowly move your head LEFT or RIGHT",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"Movement: {movement}px",
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            window_name,
            display_frame
        )

        # ----------------------------------------------------
        # Liveness threshold
        # ----------------------------------------------------

        if movement >= 35:

            movement_detected = True

            break

        frame_count += 1

        key = cv2.waitKey(30) & 0xFF

        if key == 27:

            camera.release()
            cv2.destroyAllWindows()

            return False, "Liveness check cancelled."

    camera.release()
    cv2.destroyAllWindows()

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if movement_detected:

        return (
            True,
            "🟢 Liveness verified successfully."
        )

    return (
        False,
        "🔴 Liveness verification failed. "
        "Please move your head and try again."
    )