import os
import json
import pandas as pd


PATIENT_DATA_DIR = "patient_data"


def get_patient_folder(patient_id):
    return os.path.join(
        PATIENT_DATA_DIR,
        str(patient_id)
    )


def get_profile_path(patient_id):
    return os.path.join(
        get_patient_folder(patient_id),
        "profile.json"
    )


def get_history_path(patient_id):
    return os.path.join(
        get_patient_folder(patient_id),
        "history.csv"
    )


def create_patient_profile(
    patient_id,
    name,
    age,
    gender
):

    folder = get_patient_folder(patient_id)

    os.makedirs(
        folder,
        exist_ok=True
    )

    profile = {
        "patient_id": str(patient_id),
        "name": str(name),
        "age": int(age),
        "gender": str(gender)
    }

    with open(
        get_profile_path(patient_id),
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            profile,
            file,
            indent=4
        )

    return profile


def load_patient_profile(patient_id):

    path = get_profile_path(
        patient_id
    )

    if not os.path.exists(path):
        return None

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def save_prediction(
    patient_id,
    sbp,
    dbp,
    confidence=None
):

    folder = get_patient_folder(
        patient_id
    )

    os.makedirs(
        folder,
        exist_ok=True
    )

    history_path = get_history_path(
        patient_id
    )

    result = pd.DataFrame({
        "Date": [
            pd.Timestamp.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        ],
        "SBP": [
            round(float(sbp), 2)
        ],
        "DBP": [
            round(float(dbp), 2)
        ]
    })

    if confidence is not None:

        result["Confidence"] = [
            round(float(confidence), 2)
        ]

    if os.path.exists(history_path):

        result.to_csv(
            history_path,
            mode="a",
            header=False,
            index=False
        )

    else:

        result.to_csv(
            history_path,
            index=False
        )


def load_prediction_history(patient_id):

    history_path = get_history_path(
        patient_id
    )

    if not os.path.exists(history_path):

        return pd.DataFrame(
            columns=[
                "Date",
                "SBP",
                "DBP"
            ]
        )

    return pd.read_csv(
        history_path
    )