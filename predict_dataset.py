import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

model = load_model("emotion_model.keras")
print("Modèle chargé avec succès.")

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

images_dir = "C:/Users/ayahe/Downloads/Deep Learning/Projet (HANZAZ - NCIRI - 5IIR 12)/dataset/test"

def predict_and_display_images(images_dir, num_images=10):
    image_paths = []
    for folder in os.listdir(images_dir):
        folder_path = os.path.join(images_dir, folder)
        if os.path.isdir(folder_path):  
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                image_paths.append(img_path)

    if len(image_paths) == 0:
        print("Aucune image trouvée dans le dossier spécifié.")
        return

    # Sélectionner un nombre limité d'images aléatoirement
    np.random.shuffle(image_paths)
    selected_image_paths = image_paths[:min(num_images, len(image_paths))]

    # Prétraiter les images sélectionnées
    valid_images = []
    valid_paths = []
    for img_path in selected_image_paths:
        try:
            img = load_img(img_path, target_size=(48, 48), color_mode="grayscale")
            img_array = img_to_array(img) / 255.0  
            valid_images.append(img_array)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"Impossible de lire l'image {img_path}. Elle sera ignorée.")

    if not valid_images:
        print("Aucune image valide trouvée.")
        return

    valid_images = np.array(valid_images)

    # Prédire toutes les images valides
    predictions = model.predict(valid_images)
    predicted_classes = np.argmax(predictions, axis=1)

    root = tk.Tk()
    root.title("Prédictions des émotions")

    canvas = tk.Canvas(root)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((20, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    for i, (img_path, predicted_class) in enumerate(zip(valid_paths, predicted_classes)):
        img = Image.open(img_path).resize((100, 100))  
        img_tk = ImageTk.PhotoImage(img)

        label_img = tk.Label(scrollable_frame, image=img_tk)
        label_img.image = img_tk  
        label_img.grid(row=i // 5, column=(i % 5) * 2, padx=20, pady=10)  
        label_text = tk.Label(
            scrollable_frame, 
            text=class_names[predicted_class], 
            font=("Arial", 10), 
            fg="darkblue" 
        )
        label_text.grid(row=i // 5, column=(i % 5) * 2 + 1, padx=20, pady=10)

    root.mainloop()

num_images = int(input("Entrez le nombre d'images à prédire : "))
predict_and_display_images(images_dir, num_images=num_images)
