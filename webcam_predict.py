from tensorflow.keras.models import load_model
import cv2 as cv
import numpy as np
import mediapipe as mp

model = load_model('asl_model.keras')

with open('labels.txt','r') as f:
    labels = [line.strip() for line in f.readlines()]

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands = 1,
    min_detection_confidence = 0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    frame = cv.flip(frame, 1)

    img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, c = frame.shape

            x_list = []
            y_list = []

            for lm in hand_landmarks.landmark:
                x = int(lm.x*w)
                y = int(lm.y*h)

                x_list.append(x)
                y_list.append(y)

            x_min = max(min(x_list) - 20, 0)
            x_max = min(max(x_list)  + 20, w)

            y_min = max(min(y_list) - 20, 0)
            y_max = min(max(y_list) + 20, h)

            cv.rectangle(
                    frame,
                    (x_min, y_min),
                    (x_max, y_max),
                    (0, 255, 0),
                    2
            )

            hand_crop = frame[y_min:y_max, x_min:x_max]

            img = cv.resize(hand_crop, (224, 224))

            img = img/255.0

            img = np.expand_dims(img, axis = 0)

            prediction = model.predict(img, verbose=0)

            class_index = np.argmax(prediction)

            confidence = prediction[0][class_index]

            label = labels[class_index]

            text = f'{label} ({confidence:.2f})'

            cv.putText(
                frame,
                text,
                (x_min, y_min - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv.imshow('ASL Interpreter', frame)


    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()