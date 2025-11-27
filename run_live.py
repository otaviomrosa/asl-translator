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
                    cv2.imshow('Cropped Hand', hand_crop)



        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()