import cv2
import mediapipe as mp
from mediapipe.tasks import python
import numpy as np
from pathlib import Path
import math


#initialize mediapipe config objects
model_path = Path(__file__).with_name("hand_landmarker.task")
base_options = python.BaseOptions(model_asset_path=str(model_path))
options = mp.tasks.vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

#intialize mediapipe drawing objects
mp_drawing = mp.tasks.vision.drawing_utils
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing_styles = mp.tasks.vision.drawing_styles

#landmark index pairs: (fingertip, PIP joint) for index, middle, ring, pinky
FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


#A finger is folded when its tip sits closer to the wrist than its middle joint.
#Works at any hand rotation, unlike comparing y values directly.
def is_fist(hand_landmarks):
    wrist = hand_landmarks[0]
    folded = sum(
        1 for tip, pip in FINGERS
        if _dist(hand_landmarks[tip], wrist) < _dist(hand_landmarks[pip], wrist)
    )
    return folded == 4


#Draw landmarks on hand image using mediapipe drawing utils
def draw_landmarks(img, detection_results):
    img_copy = np.copy(img)

    for hand_landmarks in detection_results.hand_landmarks:
        mp_drawing.draw_landmarks(
            img_copy,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

    return img_copy


#Detect hand landmarks using mediapipe model
def detect_hands(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    detection_result = detector.detect(img)
    drawn_image = draw_landmarks(img.numpy_view(), detection_result)

    return cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR), detection_result


#this is to read video from webcam using OpenCV
cap = cv2.VideoCapture(1)

FIST_FRAMES_NEEDED = 15   #how many frames in a row a fist must be held
fist_frames = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img, result = detect_hands(frame)

    if any(is_fist(hand) for hand in result.hand_landmarks):
        fist_frames += 1
    else:
        fist_frames = 0

    cv2.putText(img, f"fist: {fist_frames}/{FIST_FRAMES_NEEDED}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Hand Capture", img)

    key = cv2.waitKey(1)   #also gives imshow time to refresh the window

    if fist_frames >= FIST_FRAMES_NEEDED:   #close your fist to stop
        break
    if key == ord('k'):                     #press k to stop
        break

cap.release()
cv2.destroyAllWindows()