from tensorflow.keras.models import load_model
import cv2 as cv
import numpy as np

model = load_model('asl_model.keras')

with open('labels.txt','r') as f:
    labels = [line.strip() for line in f.readlines()]

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    frame = cv.flip(frame, 1)

    img = cv.resize(frame, (224, 224))

    img = img/255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    class_index = np.argmax(prediction)

    label = labels[class_index]

    confidence = np.max(prediction)

    cv.putText(frame,
               f'{label} ({confidence:.2f})',
               (20, 50),
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


