# Real-Time ASL Letter Translator 

Real-time American Sign Language (ASL) letter classifier that uses a webcam to translate static hand gestures into their corresponding letters (A-Y, excluding the motion-based 'J' and 'Z').

This project uses:
* `OpenCV` for capturing and processing the webcam feed.
* `MediaPipe` for robust, real-time hand detection and landmark tracking.
* `PyTorch` to build and train a Convolutional Neural Network (CNN) for gesture classification.
* The `Sign Language MNIST` dataset as the primary training data.