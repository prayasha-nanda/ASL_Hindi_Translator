# ASL to Hindi Translator
Real-time ASL gesture recognition system with speech output and Hindi translation. Built using MediaPipe + TensorFlow.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue"/>
  <img src="https://img.shields.io/badge/TensorFlow-orange"/>
  <img src="https://img.shields.io/badge/Streamlit-red"/>
</p>

## Demonstration

[![Watch Demo](assets/prayasha.gif)](assets/demo_video.mp4)

▶ Click the demo above to view the full video with audio. If playback doesn’t work in your browser, download the file to view it locally.

## Features

* **Real-time hand tracking** using MediaPipe
* **ASL alphabet classification** using a TensorFlow neural network
* **Prediction smoothing** using temporal buffering
* **Word & sentence formation** from predicted letters
* **Text-to-speech output** in Hindi
* **Interactive UI** built with Streamlit

## Working

1. **Capture Input**
   Webcam captures live video frames using OpenCV.

2. **Hand Landmark Detection**
   MediaPipe extracts 21 hand landmarks (x, y, z coordinates).

3. **Feature Engineering**

   * Convert landmarks into a 63-dimensional vector
   * Normalize relative to wrist position

4. **Model Prediction**

   * Feed features into a trained neural network
   * Output predicted ASL letter

5. **Post-processing**

   * Buffer predictions for stability
   * Build words and sentences

6. **Language Processing**

   * Convert ASL-style English → proper English
   * Translate to Hindi

7. **Speech Output**

   * Generate Hindi audio using gTTS

## Model Architecture

```
Input Layer: 63 features (21 landmarks × x,y,z)

Dense (128, ReLU)
Dropout (0.3)

Dense (64, ReLU)

Output Layer (Softmax)
```

* **Loss Function:** Sparse Categorical Crossentropy
* **Optimizer:** Adam
* **Accuracy:** ~76%

## Performance

* Achieved ~76% accuracy on validation data
* Strong performance on distinct gestures (F, K, R, S)
* Challenges with similar gestures (M, N, S, T)

## Limitations

* Uses **static hand landmarks only** (no motion modeling)
* Struggles with visually similar signs
* Limited grammar correction (ASL is not English syntax)

## Tech Stack

* Python
* OpenCV
* MediaPipe
* TensorFlow / Keras
* Streamlit
* deep-translator
* gTTS

## Dataset

This project uses the ASL Alphabet Dataset from Kaggle.

Contains labeled images for 26 alphabets (A–Z) along with special classes like space, del, and nothing.
Each class consists of multiple hand gesture images captured under varying conditions.

Dataset Source: https://www.kaggle.com/datasets/grassknoted/asl-alphabet

## Project Structure

```
├── assets
│   ├── demo_video.mp4      # Demo: signed sentence to Hindi speech
│   └── prayasha.gif        # GIF of author's signing
├── app.py                                         # Streamlit app
├── ASL_DL_classification_model.ipynb              # Model training script
├── asl_model.keras                                # Trained model
├── LICENSE
└── README.md
```

## All the letters

![All the ASL letters](assets/asl_letters.png)

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/prayasha-nanda/ASL_Hindi_Translator.git
cd ASL_Hindi_Translator
```

### 2. Install dependencies

```bash
pip install streamlit opencv-python numpy tensorflow mediapipe deep-translator gTTS tqdm scikit-learn

```

### 3. Run the app

```bash
streamlit run app.py
```

---

## Authors

**Prayasha Nanda**

**Hari Samhita**

---

This project is licensed under the MIT License.
