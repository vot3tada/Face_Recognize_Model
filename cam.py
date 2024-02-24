import cv2
import mediapipe as mp
from tensorflow import keras
import numpy as np
mp_face_detection = mp.solutions.face_detection
from PIL import Image, ImageFont, ImageDraw
import emoji

model = keras.models.load_model('model_dir_1/model.tf')
moods = {"0": {"mood": "Angry", "smile": ":enraged_face:", "color": (0, 0, 255)},
         "1": {"mood": "Disgust", "smile": ":nauseated_face:", "color": (0, 255, 196)},
         "2": {"mood": "Fear", "smile": ":fearful_face:", "color": (255, 0, 60)},
         "3": {"mood": "Happy", "smile": ":smiling_face_with_smiling_eyes:", "color": (109, 255, 92)},
         "4": {"mood": "Sad", "smile": ":frowning_face:", "color": (255, 255, 0)},
         "5": {"mood": "Surprise", "smile": ":face_with_open_mouth:", "color": (0, 137, 255)},
         "6": {"mood": "Neutral", "smile": ":neutral_face:", "color": (0, 255, 255)}}

#font = ImageFont.truetype("NotoEmoji-VariableFont_wght.ttf", 32)
#Put your font
font = None

cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            continue
        width = int(frame.shape[1])
        height = int(frame.shape[0])


        results = face_detection.process(frame)
        if results.detections:
            for detection in results.detections:
                x_min = int(detection.location_data.relative_bounding_box.xmin * width)
                y_min = int(detection.location_data.relative_bounding_box.ymin * height)
                x_max = int(detection.location_data.relative_bounding_box.width * width + x_min)
                y_max = int(detection.location_data.relative_bounding_box.height * height + y_min)
                if x_min < x_max and y_min < y_max:
                    face_frame = frame[y_min:y_max, x_min:x_max]
                    try:
                        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_RGB2GRAY)
                    except:
                        continue
                    face_frame = cv2.resize(face_frame, (48, 48))
                    face_frame = np.expand_dims(face_frame, axis=0)
                    p = model.predict(face_frame).tolist()[0]
                    mood = p.index(max(p))
                    mood = moods[str(mood)]
                    frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), mood['color'], 2)
                    frame = cv2.putText(frame, mood['mood'], (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, mood['color'], 2, cv2.LINE_AA)

                    # Make into PIL Image

                    if font is not None:
                        frame = Image.fromarray(frame)
                        draw = ImageDraw.Draw(frame)
                        tick = str(emoji.emojize(mood['smile']))
                        draw.text((x_min + 2, y_min + 2), tick, mood['color'], font=font)
                        frame = np.array(frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow('video', frame)
#%%

#%%

#%%
