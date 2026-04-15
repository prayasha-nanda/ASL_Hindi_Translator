import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque, Counter
from deep_translator import GoogleTranslator
from gtts import gTTS
import os
import tempfile

st.set_page_config(page_title="ASL Translator", layout="wide")

st.title("ASL to Hindi Translator")

model = tf.keras.models.load_model(
    "asl_model.keras",
    compile=False,
    safe_mode=False
)

labels = [
    "A","B","C","D","E","F","G","H","I","J",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y","Z",
    "del","nothing","space"
]

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

def extract_keypoints(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    keypoints = np.zeros(63)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        base_x = hand.landmark[0].x
        base_y = hand.landmark[0].y
        base_z = hand.landmark[0].z

        for i, lm in enumerate(hand.landmark):

            keypoints[i*3]   = lm.x - base_x
            keypoints[i*3+1] = lm.y - base_y
            keypoints[i*3+2] = lm.z - base_z

        max_val = np.max(np.abs(keypoints))
        if max_val != 0:
            keypoints = keypoints / max_val

    return keypoints, results

if "word" not in st.session_state:
    st.session_state.word = ""

if "sentence" not in st.session_state:
    st.session_state.sentence = ""

if "buffer" not in st.session_state:
    st.session_state.buffer = deque(maxlen=7)

if "last_letter" not in st.session_state:
    st.session_state.last_letter = ""

col1, col2 = st.columns(2)

run = st.checkbox("Start Camera")

FRAME_WINDOW = col1.image([])
info_box = col2.empty()

cap = cv2.VideoCapture(0)

confidence_threshold = 0.80

while run:

    ret, frame = cap.read()
    if not ret:
        st.error("Camera not accessible")
        break

    frame = cv2.flip(frame, 1)

    keypoints, results = extract_keypoints(frame)

    input_data = np.expand_dims(keypoints, axis=0)

    pred = model.predict(input_data, verbose=0)[0]

    idx = np.argmax(pred)
    confidence = np.max(pred)

    letter = labels[idx]

    if confidence > confidence_threshold:

        if letter == "nothing":
            pass

        elif letter == "space":
            if st.session_state.word != "":
                st.session_state.sentence += st.session_state.word + " "
                st.session_state.word = ""

            st.session_state.buffer.clear()
            st.session_state.last_letter = ""

        elif letter == "del":
            st.session_state.word = st.session_state.word[:-1]
            st.session_state.buffer.clear()
            st.session_state.last_letter = ""

        else:
            st.session_state.buffer.append(letter)
            stable = Counter(st.session_state.buffer).most_common(1)[0][0]

            if stable != st.session_state.last_letter:
                st.session_state.word += stable
                st.session_state.last_letter = stable

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


    FRAME_WINDOW.image(frame, channels="BGR")

    info_box.markdown(f"""
    ### Live Output

    **Letter Buffer Word:** `{st.session_state.word}`

    **Sentence:** `{st.session_state.sentence}`

    **Current Letter:** `{letter}`  
    **Confidence:** `{confidence:.2f}`
    """)

cap.release()

st.markdown("---")

if st.button("Translate to Hindi"):

    if st.session_state.sentence.strip() != "":

        hindi = GoogleTranslator(
            source='auto',
            target='hi'
        ).translate(st.session_state.sentence)

        st.success(f"English: {st.session_state.sentence}")
        st.info(f"Hindi: {hindi}")

        tts = gTTS(hindi, lang='hi')

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp_file.name)

        st.audio(tmp_file.name)

    else:
        st.warning("No sentence to translate")

if st.button("Reset"):
    st.session_state.word = ""
    st.session_state.sentence = ""
    st.session_state.buffer.clear()
    st.session_state.last_letter = ""
    st.success("Reset done")
