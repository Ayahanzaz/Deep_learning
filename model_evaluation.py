import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = load_model("emotion_model.keras")

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

test_dir = "C:/Users/ayahe/Downloads/Deep Learning/Projet (HANZAZ - NCIRI - 5IIR 12)/dataset/test"

image_arrays = []  
true_labels = []   

for class_index, class_folder in enumerate(os.listdir(test_dir)):
    folder_path = os.path.join(test_dir, class_folder)
    if os.path.isdir(folder_path):
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            img = load_img(img_path, target_size=(48, 48), color_mode="grayscale")
            img_array = img_to_array(img) / 255.0
            image_arrays.append(img_array)
            true_labels.append(class_index)

image_arrays = np.array(image_arrays)
true_labels = np.array(true_labels)

predictions = model.predict(image_arrays)
predicted_labels = np.argmax(predictions, axis=1)

true_labels_names = [class_names[label] for label in true_labels]
predicted_labels_names = [class_names[label] for label in predicted_labels]

conf_matrix = confusion_matrix(true_labels_names, predicted_labels_names, labels=class_names)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Matrice de confusion")
plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.show()

print("Rapport de classification :")
print(classification_report(true_labels_names, predicted_labels_names, target_names=class_names))
