import serial
import numpy as np
import cv2

# Настройки последовательного порта
ser = serial.Serial('COM5', 115200) # Замените 'COM3' на ваш порт

# Размеры изображения (QVGA)
width = 320
height = 240

while True:
    # Ждём поступления данных
    while ser.in_waiting < width * height * 2:
        pass

    # Считываем данные с камеры (RGB565)
    data = ser.read(width * height * 2)

    # Преобразуем байты в массив numpy (16-битные пиксели)
    img = np.frombuffer(data, dtype=np.uint16).reshape((height, width))

    # Отобразим изображение (для теста можно оставить как есть)
    cv2.imshow('Camera Image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ser.close()
cv2.destroyAllWindows()
