import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model('sign_language_model.h5')

# Dictionary to map label indices to sign language letters
label_map = {0: 'A', 1: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P'}

# Function to preprocess the frame
def preprocess_frame(frame):
    # Resize frame to match the input size of the model
    frame = cv2.resize(frame, (64, 64))
    # Normalize pixel values to range [0, 1]
    frame = frame / 255.0
    # Expand dimensions to match the input shape expected by the model
    frame = np.expand_dims(frame, axis=0)
    return frame

# Function to perform inference on a frame
def predict_sign(frame):
    # Preprocess the frame
    frame = preprocess_frame(frame)
    # Perform inference
    prediction = model.predict(frame)
    # Get the predicted label index
    predicted_label_index = np.argmax(prediction)
    # Get the confidence score for the predicted label
    confidence_score = prediction[0][predicted_label_index]
    # Map the label index to the corresponding sign language letter
    predicted_letter = label_map[predicted_label_index]
    return predicted_letter, confidence_score

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

# Loop to capture and process frames from the webcam
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Perform sign language detection on the frame
    predicted_letter, confidence_score = predict_sign(frame)

    # Display the predicted letter and confidence score on the frame
    text = f'{predicted_letter} ({confidence_score*100:.2f})'
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Sign Language Detection', frame)

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
