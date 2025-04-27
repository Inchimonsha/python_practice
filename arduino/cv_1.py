import serial
import numpy as np
import cv2

# Настройка последовательного порта (замените COM3 на ваш порт)
ser = serial.Serial('COM5', 115200)

# Размеры изображения (QVGA)
width = 140
height = 100

def rgb565_to_bgr888(frame_rgb565):
    frame_rgb565 = np.frombuffer(frame_rgb565, dtype=np.uint8).reshape((height, width, 2))
    r = ((frame_rgb565[:, :, 0] & 0xF8) >> 3) * 8
    g = (((frame_rgb565[:, :, 0] & 0x07) << 3) | ((frame_rgb565[:, :, 1] & 0xE0) >> 5)) * 4
    b = (frame_rgb565[:, :, 1] & 0x1F) * 8
    frame_bgr = np.stack((b, g, r), axis=-1)
    return frame_bgr

while True:
    # Чтение одного кадра (QVGA: ширина * высота * 2 байта на пиксель)
    frame_data = ser.read(width * height * 2)

    # Преобразование данных в массив NumPy
    frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 2))

    # Преобразование из RGB565 в BGR888 для OpenCV
    # bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB5652BGR)
    frame_bgr = rgb565_to_bgr888(frame_data)

    # Отображение изображения с камеры
    cv2.imshow("Camera Stream", frame_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ser.close()
cv2.destroyAllWindows()
