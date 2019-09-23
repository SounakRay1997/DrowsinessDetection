'''This project is designed to perform real time drowsiness detection of drivers. Many existing drowsiness detection systems use only the aspect ratio of the eye to detect drowsiness. The problem with this is that if the driver already closes his eyes before the alarm goes off, the chances of an accident increases by many times. Hence, here I have also used the aspect ratio of the mouth to detect if a driver is yawning or not. Yawning is usually a pre-cursor to drowsiness. So we also set an alarm if the driver yawns to ensure his full attention. In this way we could also reduce the chances of an accident drastically.
'''

import cv2
import dlib
import imutils
import time
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import playsound

def play_alarm(alarm_path):
    playsound.playsound(alarm_path)
    
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    eye_width = dist.euclidean(eye[0], eye[3])
    eye_height=(A+B)/2
    aspect_ratio=eye_height/eye_width
    return aspect_ratio
	
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[12], mouth[18])
    B = dist.euclidean(mouth[13], mouth[17])
    C = dist.euclidean(mouth[14], mouth[16])
    mouth_height = (A+B+C)/3
    mouth_width = dist.euclidean(mouth[11], mouth[15])
    aspect_ratio = mouth_height/mouth_width
    return aspect_ratio
	
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-e", "--eye_alarm", type=str, default="", help="path eye alarm .WAV file")
ap.add_argument("-m", "--mouth_alarm", type=str, default="", help="path mouth alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
args = vars(ap.parse_args())


EYE_THRESHOLD_VALUE=0.25
EYE_NUMBER_OF_CONSECUTIVE_FRAMES=40

EYE_FRAME_COUNTER=0
EYE_SET_ALARM=False

MOUTH_THRESHOLD_VALUE=0.7
MOUTH_NUMBER_OF_CONSECUTIVE_FRAMES=30

MOUTH_FRAME_COUNTER=0
MOUTH_SET_ALARM=False

face_detector = dlib.get_frontal_face_detector()
landmarks_predictor = dlib.shape_predictor(args["shape_predictor"])

(leftStart, leftEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightStart, rightEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(start, end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

while True:
    frame = vs.read()
    print(frame)
    print(frame.shape)
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    faces = face_detector(gray, 0)
    for face in faces:
        facial_points = landmarks_predictor(gray, face)
        facial_points = face_utils.shape_to_np(facial_points)
        leftEye = facial_points[leftStart:leftEnd]
        rightEye = facial_points[rightStart:rightEnd]
        mouth = facial_points[start:end]
		
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        mouthAR = mouth_aspect_ratio(mouth)
		
        eyeAR = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
		
        if eyeAR < EYE_THRESHOLD_VALUE:
            EYE_FRAME_COUNTER += 1
            if EYE_FRAME_COUNTER >= EYE_NUMBER_OF_CONSECUTIVE_FRAMES:
                if not EYE_SET_ALARM:
                    EYE_SET_ALARM = True
                    if args["eye_alarm"] != "":
                        t = Thread(target=play_alarm, args=(args["eye_alarm"],))
                        t.deamon = True
                        t.start()
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            EYE_FRAME_COUNTER = 0
            EYE_SET_ALARM = False
            cv2.putText(frame, "eyeAR: {:.2f}".format(eyeAR), (280, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			
        if mouthAR < MOUTH_THRESHOLD_VALUE:
            MOUTH_FRAME_COUNTER += 1
            if MOUTH_FRAME_COUNTER >= MOUTH_NUMBER_OF_CONSECUTIVE_FRAMES:
                if not MOUTH_SET_ALARM:
                    MOUTH_SET_ALARM = True
                    if args["mouth_alarm"] != "":
                        t1 = Thread(target=play_alarm, args=(args["mouth_alarm"],))
                        t1.deamon = True
                        t1.start()
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            MOUTH_FRAME_COUNTER = 0
            MOUTH_SET_ALARM = False
            cv2.putText(frame, "mouthAR: {:.2f}".format(mouthAR), (280, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
            
cv2.destroyAllWindows()
vs.stop()
