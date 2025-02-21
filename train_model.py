import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Chemins vers les dossiers
train_dir = "C:/Users/ayahe/Downloads/Deep Learning/Projet (HANZAZ - NCIRI - 5IIR 12)/dataset/train"
test_dir = "C:/Users/ayahe/Downloads/Deep Learning/Projet (HANZAZ - NCIRI - 5IIR 12)/dataset/test"

if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Le dossier d'entraînement est introuvable : {train_dir}")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Le dossier de test est introuvable : {test_dir}")

# Générateurs de données avec augmentation pour réduire l'overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,  # Augmentation de la taille de batch pour plus de stabilité
    color_mode="grayscale",
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical"
)

# Modèle CNN amélioré
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(512, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dropout(0.5),  
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  
])

# Compilation avec optimizer ajusté
model.compile(
    optimizer=Adam(learning_rate=0.0005),  # Augmentation initiale pour accélérer la convergence
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks pour optimiser l'entraînement
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
]

# Entraînement
print("Début de l'entraînement...")
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=test_generator,
    callbacks=callbacks

    
)
print("Entraînement terminé.")

# Sauvegarde du modèle
model.save("emotion_model.keras")
print("Modèle sauvegardé avec succès sous le nom 'emotion_model.keras'.")
