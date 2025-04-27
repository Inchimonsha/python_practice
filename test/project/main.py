import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
import os

# Загрузка предварительно обученной модели VGG16
base_model = tf.keras.applicationsю.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Замораживание слоев базовой модели
for layer in base_model.layers:
    layer.trainable = False

# Добавление пользовательских слоев
model = tf.keras.models.Sequential()
model.add(base_model)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='softmax'))

# Компиляция модели
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Функция для загрузки данных из XML-файлов
def load_data_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Извлечение информации об объектах из XML-файла
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        objects.append((name, xmin, ymin, xmax, ymax))

    return objects


# Загрузка и подготовка данных
train_images = []
train_labels = []
train_bboxes = []

for filename in os.listdir('train_images'):
    if filename.endswith('.jpg'):
        image_path = os.path.join('train_images', filename)
        xml_path = os.path.join('train_annotations', os.path.splitext(filename)[0] + '.xml')

        # Загрузка изображения
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        train_images.append(tf.keras.preprocessing.image.img_to_array(image))

        # Загрузка разметки из XML-файла
        objects = load_data_from_xml(xml_path)
        train_labels.append([obj[0] for obj in objects])
        train_bboxes.append([obj[1:] for obj in objects])

train_images = np.array(train_images)
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=1)

# Обучение модели
model.fit(train_images, train_labels,
          validation_data=(val_images, val_labels),
          epochs=50,
          batch_size=32)

# Сохранение обученной модели в формате XML
model_xml = model.to_xml()
with open('model.xml', 'w') as f:
    f.write(model_xml)