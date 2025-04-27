import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import tensorflow as tf

# Укажите путь к папке с изображениями
image_dir = 'path/to/images'

# Укажите путь к папке с XML файлами
xml_dir = 'path/to/xml'

# Создаем список изображений и соответствующих им XML файлов
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]

# Создаем списки для хранения изображений и меток
images = []
labels = []

# Загружаем изображения и извлекаем метки из XML файлов
for image_file, xml_file in zip(image_files, xml_files):
    # Загружаем изображение
    image = Image.open(os.path.join(image_dir, image_file))
    image = image.resize((224, 224))  # Изменяем размер изображения до 224x224
    images.append(np.array(image))

    # Извлекаем метки из XML файла
    tree = ET.parse(os.path.join(xml_dir, xml_file))
    root = tree.getroot()
    label = [int(obj.find('name').text) for obj in root.findall('object')]
    labels.append(label)

# Преобразуем списки в массивы NumPy
images = np.array(images)
labels = np.array(labels)

# Разделяем данные на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

# Укажите путь к папке с изображениями
image_dir = r'path/to/images'

# Укажите путь к папке с XML файлами
xml_dir = 'path/to/xml'

# Создаем список изображений и соответствующих им XML файлов
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]

# Создаем списки для хранения изображений и меток
images = []
labels = []

# Загружаем изображения и извлекаем метки из XML файлов
for image_file, xml_file in zip(image_files, xml_files):
    # Загружаем изображение
    image = Image.open(os.path.join(image_dir, image_file))
    image = image.resize((224, 224))  # Изменяем размер изображения до 224x224
    images.append(np.array(image))

    # Извлекаем метки из XML файла
    tree = ET.parse(os.path.join(xml_dir, xml_file))
    root = tree.getroot()
    label = [int(obj.find('name').text) for obj in root.findall('object')]
    labels.append(label)

# Преобразуем списки в массивы NumPy
images = np.array(images)
labels = np.array(labels)

# Разделяем данные на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Загружаем предварительно обученную модель VGG16
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Замораживаем веса базовой модели
for layer in base_model.layers:
    layer.trainable = False

# Добавляем дополнительные слои
model = tf.keras.models.Sequential()
model.add(base_model)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(np.unique(labels)), activation='sigmoid'))

# Компилируем модель
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Увеличиваем данные для улучшения обобщения
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# Обучаем модель
model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=50,
    validation_data=(X_test, y_test),
    steps_per_epoch=len(X_train) // 32,
    verbose=1)

model.save('model.xml')