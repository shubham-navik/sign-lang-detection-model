import cv2
import os

# Define the directory to save the dataset
directory = './Dataset'

# Create the main directory if it doesn't exist
if not os.path.exists(directory):
    os.mkdir(directory)

# Create directories for each letter and the blank category
for i in range(65, 91):  # ASCII values for A-Z
    letter = chr(i)
    if not os.path.exists(os.path.join(directory, letter)):
        os.mkdir(os.path.join(directory, letter))

if not os.path.exists(os.path.join(directory, 'blank')):
    os.mkdir(os.path.join(directory, 'blank'))

# Open the webcam
cap = cv2.VideoCapture(0)

# Loop to capture images
while True:
    _, frame = cap.read()
    count = {
        'a': len(os.listdir(os.path.join(directory, "A"))),
        'b': len(os.listdir(os.path.join(directory, "B"))),
        'c': len(os.listdir(os.path.join(directory, "C"))),
        'd': len(os.listdir(os.path.join(directory, "D"))),
        'e': len(os.listdir(os.path.join(directory, "E"))),
        'f': len(os.listdir(os.path.join(directory, "F"))),
        'g': len(os.listdir(os.path.join(directory, "G"))),
        'h': len(os.listdir(os.path.join(directory, "H"))),
        'i': len(os.listdir(os.path.join(directory, "I"))),
        'j': len(os.listdir(os.path.join(directory, "J"))),
        'k': len(os.listdir(os.path.join(directory, "K"))),
        'l': len(os.listdir(os.path.join(directory, "L"))),
        'm': len(os.listdir(os.path.join(directory, "M"))),
        'n': len(os.listdir(os.path.join(directory, "N"))),
        'o': len(os.listdir(os.path.join(directory, "O"))),
        'p': len(os.listdir(os.path.join(directory, "P"))),
        'q': len(os.listdir(os.path.join(directory, "Q"))),
        'r': len(os.listdir(os.path.join(directory, "R"))),
        's': len(os.listdir(os.path.join(directory, "S"))),
        't': len(os.listdir(os.path.join(directory, "T"))),
        'u': len(os.listdir(os.path.join(directory, "U"))),
        'v': len(os.listdir(os.path.join(directory, "V"))),
        'w': len(os.listdir(os.path.join(directory, "W"))),
        'x': len(os.listdir(os.path.join(directory, "X"))),
        'y': len(os.listdir(os.path.join(directory, "Y"))),
        'z': len(os.listdir(os.path.join(directory, "Z"))),
        'blank': len(os.listdir(os.path.join(directory, "blank")))
    }

    row = frame.shape[1]
    col = frame.shape[0]
    cv2.rectangle(frame, (0, 40), (300, 300), (255, 255, 255), 2)
    cv2.imshow("data", frame)
    frame = frame[40:300, 0:300]
    cv2.imshow("ROI", frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (64, 64))
    interrupt = cv2.waitKey(10)

    # Save images based on key press
    key_press = chr(interrupt & 0xFF)
    if key_press in count.keys():
        cv2.imwrite(os.path.join(directory, key_press, str(count[key_press])) + '.jpg', frame)
    elif key_press == '.':
        cv2.imwrite(os.path.join(directory, 'blank', str(count['blank'])) + '.jpg', frame)

    # Exit loop if 'q' key is pressed
    if key_press == 'q':
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
