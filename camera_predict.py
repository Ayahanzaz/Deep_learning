import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("emotion_model.keras")

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

emotion_colors = {
    'Angry': (0, 0, 255),       
    'Disgust': (0, 128, 0),     
    'Fear': (128, 0, 128),      
    'Happy': (0, 255, 0),      
    'Sad': (255, 0, 0),         
    'Surprise': (0, 255, 255), 
    'Neutral': (192, 192, 192) 
}

emotion_quotes = {
    'Angry': "La colere est un poison que tu bois en esperant que l'autre en souffre. Lache prise, libere-toi.",
    'Disgust': "Meme dans l'obscurité, il y a de la lumiere. Cherche la beaute dans chaque situation.",
    'Fear': "Affronte tes peurs, elles disparaitront. Chaque victoire commence par un pas.",
    'Happy': "Garde ce sourire, il illumine le monde autour de toi.",
    'Sad': "Apres la pluie vient toujours le beau temps. Chaque larme prepare un sourire a venir.",
    'Surprise': "Les surprises sont les epices de la vie. Accueille-les avec curiosite.",
    'Neutral': "La paix interieure commence quand tu acceptes le moment present tel qu'il est."
}


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire l'image depuis la caméra.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray_frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))
        face_array = np.expand_dims(np.expand_dims(face_resized, axis=-1), axis=0) / 255.0

        predictions = model.predict(face_array, verbose=0)
        predicted_class = class_names[np.argmax(predictions)]
        color = emotion_colors[predicted_class] 

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.putText(frame, predicted_class, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

        quote = emotion_quotes[predicted_class]
        cv2.putText(frame, quote, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Emotion Detection - Appuie sur 'q' pour quitter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
