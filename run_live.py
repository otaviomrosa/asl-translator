"""
https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
https://stackoverflow.com/questions/66876906/create-a-rectangle-around-all-the-points-returned-from-mediapipe-hand-landmark-d
https://opencv.org/blog/cropping-an-image-using-opencv/
"""

#importthelibrary
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



# J and Z are not included in the dataset because they require motion
LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

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
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_scale = clahe.apply(gray_scale)

    transform = transforms.Compose([ # Data augmentation and normalization
        transforms.ToPILImage(), # Convert numpy to PIL
        transforms.Resize((28, 28)),
        transforms.ToTensor(),   # Converts to Tensor and scales to [0, 1]
    ])

    hand = transform(gray_scale)
    
    # Debug: visualize what the model sees
    debug_img = hand.squeeze(0).numpy() # 28x28
    # Scale up to 280x280 for visibility
    debug_img_display = cv2.resize(debug_img, (280, 280), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('Model Input', debug_img_display)

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
text_to_draw = None # Initialize outside the loop to persist across frames

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
        
        # Create a copy for drawing annotations so we can crop from the clean image
        image_display = image.copy()
        
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
                
                # Draw on the display image, NOT the original image used for cropping
                cv2.rectangle(image_display, (max(0, x_min - 50), max(0, y_min - 50)), (min(w, x_max + 50), min(h, y_max + 50)), (0, 255, 0), 2)
                mp_drawing.draw_landmarks(image_display, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Calculate bounding box dimensions
                box_w = x_max - x_min
                box_h = y_max - y_min
                
                # Make the crop square to avoid distortion when resizing to 28x28
                max_side = max(box_w, box_h)
                
                # Add a margin (proportional to the hand size)
                margin = int(max_side * 0.2) # 20% margin
                side = max_side + 2 * margin
                
                # Center of the hand
                center_x = x_min + box_w // 2
                center_y = y_min + box_h // 2
                
                # Calculate crop coordinates
                x_start = center_x - side // 2
                y_start = center_y - side // 2
                x_end = x_start + side
                y_end = y_start + side
                
                # Ensure coordinates are within image bounds
                x_start_clamped = max(0, x_start)
                y_start_clamped = max(0, y_start)
                x_end_clamped = min(w, x_end)
                y_end_clamped = min(h, y_end)
                
                # Crop the hand region from the CLEAN image
                hand_crop = image[y_start_clamped:y_end_clamped, x_start_clamped:x_end_clamped]
                
                # Pad if necessary to maintain square aspect ratio (avoid stretching)
                if hand_crop.size != 0:
                    pad_top = max(0, y_start_clamped - y_start)
                    pad_bottom = max(0, y_end - y_end_clamped)
                    pad_left = max(0, x_start_clamped - x_start)
                    pad_right = max(0, x_end - x_end_clamped)
                    
                    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                        # Use border replication or constant color
                        hand_crop = cv2.copyMakeBorder(hand_crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[128, 128, 128])

                # Optionally, show cropped hand in a separate window
                if hand_crop.size != 0:
                    # Skip if crop is too small
                    if hand_crop.shape[0] < 20 or hand_crop.shape[1] < 20:
                        continue
                        
                    curr = time.time()
                    if curr - prev >= delay:
                        # predicting the letter
                        idx_l, conf = prediction(hand_crop, model, device)
                        prev = curr
                    
                        # if confidence is high enough --> show letter
                        if conf > 0.5:
                            text_to_draw = f"{idx_l} ({conf:.2f})"
                        else:
                            print("confidence is too low")
                    

                    cv2.imshow('Cropped Hand', hand_crop)

        # resizing for smaller live window
        # Flip the image horizontally for a selfie-view display
        image_display = cv2.flip(image_display, 1)
        
        # Draw the prediction text if available (after flipping so text isn't mirrored)
        if text_to_draw:
            cv2.putText(image_display, text_to_draw, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3, cv2.LINE_AA)

        # resizing for smaller live window
        sizing = 0.5
        image_display = cv2.resize(image_display, None, fx = sizing, fy = sizing, interpolation = cv2.INTER_LINEAR)
        cv2.imshow('MediaPipe Hands', image_display)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()