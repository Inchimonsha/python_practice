import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm

# Загрузка и подготовка данных
coco_dir = r'C:\main\lang\python\test\project\file_project'
train_dir = os.path.join(coco_dir, 'train')
val_dir = os.path.join(coco_dir, 'val')

# Определение размера изображений
img_size = (224, 224)

# Загрузка изображений и меток
X_train = []
y_train = []
for subdir in os.listdir(train_dir):
    class_dir = os.path.join(train_dir, subdir)
    for filename in os.listdir(class_dir):
        img_path = os.path.join(class_dir, filename)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        X_train.append(img_array)
        y_train.append(int(subdir))

X_train = np.array(X_train)
y_train = np.array(y_train)

X_val = []
y_val = []
for subdir in os.listdir(val_dir):
    class_dir = os.path.join(val_dir, subdir)
    for filename in os.listdir(class_dir):
        img_path = os.path.join(class_dir, filename)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        X_val.append(img_array)
        y_val.append(int(subdir))

X_val = np.array(X_val)
y_val = np.array(y_val)

# Построение модели
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = tf.keras.Sequential()
model.add(base_model)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(os.listdir(train_dir)), activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
epochs = 50
batch_size = 32

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)