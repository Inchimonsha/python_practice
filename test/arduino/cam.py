import serial
import numpy as np
import cv2

# Откройте последовательный порт
ser = serial.Serial('COM3', 9600)  # Укажите правильный порт

# Создайте буфер для хранения данных
buffer = bytearray()

while True:
    # Читайте данные из последовательного порта
    data = ser.read(1)
    if data:
        buffer.extend(data)

    # Если буфер полон (QVGA изображение)
    if len(buffer) >= 320 * 240:
        # Преобразуйте буфер в numpy массив
        img = np.frombuffer(buffer, dtype=np.uint8).reshape(240, 320)

        # Отобразите изображение
        cv2.imshow('Image', img)

        # Очистите буфер
        buffer = bytearray()

        # Выход по нажатию клавиши
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Закройте последовательный порт и окно
ser.close()
cv2.destroyAllWindows()
