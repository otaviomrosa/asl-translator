"""
https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
https://stackoverflow.com/questions/66876906/create-a-rectangle-around-all-the-points-returned-from-mediapipe-hand-landmark-d
https://opencv.org/blog/cropping-an-image-using-opencv/
"""

#import the library
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import torch
import numpy as np
from src.model import ASLClassifier
from torchvision import transforms
import time



# Need to Skip J & Z just temp for now
LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "*J"
           "I", "K", "L", "M", "N", "O", "P",
           "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "*Z"]

# Loading training model
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASLClassifier()
    model.load_state_dict(torch.load('models/asl_model.pth', map_location = device))
    model.eval()
    return model, device




# Predict the letter from the hand image
def prediction(handcrop, model, device):
    print("prediction called") # debug
    gray_scale = cv2.cvtColor(handcrop, cv2.COLOR_BGR2GRAY)

    transform = transforms.Compose([ # Data augmentation and normalization
        transforms.ToPILImage(), # Convert numpy to PIL
        transforms.Resize((28, 28)),
        transforms.ToTensor(),   # Converts to Tensor and scales to [0, 1]
    ])

    hand = transform(gray_scale)
    hand= hand.unsqueeze(0) # add batch dimension
    hand= hand.to(device)

    with torch.no_grad():
        raw_num = model(hand)
        prob_num = torch.softmax(raw_num, dim=1)
        predict, index = prob_num.max(dim=1) # predict = confidence level, letter = index

    pr = predict.item()
    idx_l = index.item()
    print(f"Predicted: {LETTERS[idx_l]}, Confidence: {pr:.2f}")
    return LETTERS[idx_l], pr


model, device = load_model()

prev = 0
delay = 1 # predict every 1 second



#Template from mediapipe doc
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        #image = cv2.flip(image, 1)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        h, w, c = image.shape


        # Draw the hand annotations on the image.
        #image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                cv2.rectangle(image, (x_min - 50, y_min - 50), (x_max + 50, y_max + 50), (0, 255, 0), 2)
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                x_min_crop = max(x_min, 0)
                y_min_crop = max(y_min, 0)
                x_max_crop = min(x_max, w)
                y_max_crop = min(y_max, h)

                # Crop the hand region from the image
                hand_crop = image[y_min_crop - 50:y_max_crop + 50, x_min_crop - 50:x_max_crop + 50]
                
                # Optionally, show cropped hand in a separate window
                if hand_crop.size != 0:
                    curr = time.time()
                    if curr - prev >= delay:
                        # predicting the letter
                        idx_l, conf = prediction(hand_crop, model, device)
                        prev = curr
                    # if confidence is high enough --> show letter
                    if conf > 0.5:
                        cv2.putText(image, f"{idx_l} ({conf:.2f})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3, 
                                    cv2.LINE_AA)
                    else:
                        print("confidence is too low")
                    

                    cv2.imshow('Cropped Hand', hand_crop)

        # resizing for smaller live window
        sizing = 0.5
        image = cv2.resize(image, None, fx = sizing, fy = sizing, interpolation = cv2.INTER_LINEAR)
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()