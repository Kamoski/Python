import cv2
import pandas as pd
from matplotlib import pyplot as plt

def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    smiles_number = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        smiles_number += len(smiles)

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
    return frame, smiles_number

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_smile.xml')


video_capture = cv2.VideoCapture(0)
smiles_numbers = []
miliseconds_marks = []
while video_capture.isOpened():
# Captures video_capture frame by frame
    frame_exists, frame = video_capture.read() 

    if frame_exists:
        # To capture image in monochrome                    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calls the detect() function    
        canvas, amount_of_smiles = detect(gray, frame)   
        smiles_numbers.append(amount_of_smiles)
        miliseconds_marks.append(video_capture.get(cv2.CAP_PROP_POS_MSEC))
        # Displays the result on camera feed                     
        cv2.imshow('Video', frame) 

        # The control breaks once q key is pressed                        
        if cv2.waitKey(1) & 0xff == ord('q'):               
            break
    else:
        cv2.destroyAllWindows()
        break

# Release the capture once all the processing is done.
video_capture.release()                                 
cv2.destroyAllWindows()

df = pd.DataFrame({"x":miliseconds_marks,
                   "y":smiles_numbers})
df.to_csv("smiling3_data.csv", sep='\t')