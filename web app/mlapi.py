from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import base64
import numpy as np
from mediapipe.python.solutions import holistic as mp_holistic
import mediapipe.python.solutions.drawing_utils as mp_drawing
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
from mediapipe.python.solutions.holistic import Holistic
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False        
    results = model.process(image)
    image.flags.writeable = True   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(
    image, 
    results.left_hand_landmarks, 
    mp_holistic.HAND_CONNECTIONS,
    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=4),  # Циан
    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2),  # Желтый
    )
    mp_drawing.draw_landmarks(
        image, 
        results.right_hand_landmarks, 
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=4),  # Маджента
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),    # Зеленый
    )

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

actions = np.array(['zero', 'one', 'five'])
threshold = 0.8

class ImageData(BaseModel):
    image: str


model = load_model('action.h5')

@app.post('/predict')
async def predict(data: ImageData):
    try:
        header, encoded = data.image.split(",", 1) if "," in data.image else ("", data.image)
        decoded_data = base64.b64decode(encoded)
        np_data = np.frombuffer(decoded_data, np.uint8)
        image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

        holistic = Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        image, results = mediapipe_detection(image, holistic)
        
        
        print('  \t sd;fjsdkf')
        draw_styled_landmarks(image, results)
        print('  \t ызуывывдоат')
        
        
        keypoints = extract_keypoints(results)
        print(results, '  \t results')
        sequence = [keypoints] * 30



        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        print(actions[np.argmax(res)], ' \t actions[np.argmax(res)],')
        
        if res[np.argmax(res)] > threshold:
            print("Predicted action: ", actions[np.argmax(res)])
            return actions[np.argmax(res)]
        else:
            print("No significant action detected.")
            return "No significant action detected."
    except Exception as e:
        print('-------------------')
        raise HTTPException(status_code=400, detail=str(e))