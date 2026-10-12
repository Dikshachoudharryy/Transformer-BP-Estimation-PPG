# Transformer-Based Cuffless Blood Pressure Estimation Using PPG

A research-oriented project exploring cuffless blood pressure estimation from photoplethysmography (PPG) signals using Transformer-based deep learning and Explainable AI (XAI).

## Overview

Conventional blood pressure measurement often requires a cuff. This project investigates an approach to estimating systolic blood pressure (SBP) and diastolic blood pressure (DBP) from PPG signals using signal processing, deep learning, and model interpretability techniques.

**Project status:** Research prototype. Clinical validity and real-world performance require further evaluation.

## Objectives

- Process and validate PPG signal data.
- Investigate Transformer-based models for blood pressure estimation.
- Estimate SBP and DBP from input signals.
- Explore model interpretability using Explainable AI techniques.
- Provide a Python-based inference and visualization interface.

## Technology Stack

- **Language:** Python
- **Deep Learning:** Transformer-based architecture
- **Signal Processing:** PPG preprocessing and scalogram-based analysis, where applicable
- **Explainable AI:** Grad-CAM and other implemented interpretability methods
- **Interface:** Python application (`app.py`)
- **Analysis:** Jupyter Notebook

## Project Structure

```text
VT_Project/
├── app.py
├── model.py
├── utils.py
├── patient_data.py
├── xai.py
├── verification.py
├── test_v1_inference.py
├── 03_XAI_GradCAM.ipynb
├── Dataset_Validation.ipynb
├── Transformer_Training.ipynb
├── Transformer_XAI_Cuffless_BP.ipynb
├── Webcam.ipynb
├── notebooks/
├── .gitignore
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Dikshachoudharryy/Transformer-BP-Estimation-PPG.git
cd Transformer-BP-Estimation-PPG
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the packages required by the project. If a `requirements.txt` file is available, run:

```bash
pip install -r requirements.txt
```

### 4. Run the application

Check the application's entry point and required configuration, then run:

```bash
python app.py
```

The exact setup may depend on the model weights, dataset files, and packages used by the current implementation.

## Explainability

The project includes an XAI workflow for investigating which signal regions or features influence model predictions. Interpretability outputs should be treated as model-analysis aids, not proof of physiological causation.

## Dataset and Privacy

This repository does not include the datasets or model checkpoints excluded by `.gitignore`. Obtain any required data through its official source and follow its access, licensing, and privacy requirements.

Do not upload restricted patient-level data or identifiable medical information.

## Evaluation

Evaluate the model using appropriate regression metrics, such as:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Error and standard deviation of error

Report actual measured results here only after verifying them on a held-out test set. Avoid making clinical accuracy claims without appropriate validation.

## Limitations

This is a research prototype and is not a substitute for a clinically validated blood pressure monitor. Predictions should not be used for diagnosis or treatment decisions.

## Future Work

- Improve signal-quality assessment and preprocessing.
- Validate performance across independent subjects and datasets.
- Evaluate robustness to motion artifacts and changes in measurement conditions.
- Improve interpretability and reproducibility.
- Investigate requirements for clinical validation.

## Author

**Diksha Choudhary**

GitHub: [Dikshachoudharryy](https://github.com/Dikshachoudharryy)