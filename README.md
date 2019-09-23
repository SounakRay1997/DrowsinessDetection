# DrowsinessDetection

This project is designed to perform real time drowsiness detection of drivers. Many existing drowsiness detection systems use only the aspect ratio of the eye to detect drowsiness. The problem with this is that if the driver already closes his eyes before the alarm goes off, the chances of an accident increases by many times. Hence, here I have also used the aspect ratio of the mouth to detect if a driver is yawning or not. Yawning is usually a pre-cursor to drowsiness. So we also set an alarm if the driver yawns to ensure his full attention. In this way we could also reduce the chances of an accident drastically.


# How to use the code

```python DrowsinessDetection.py --shape-predictor shape_predictor_68_face_landmarks.dat --eye_alarm eye_closing_alarm.mp3 --mouth_alarm mouth_closing_alarm.mp3```

The shape_predictor_68_face_landmarks.dat file is a trained file to detect the facial landmarks in a frame. It is available [here](https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat). 
