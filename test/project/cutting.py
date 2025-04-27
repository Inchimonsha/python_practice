import cv2
import time

def cut_video():
    # Открываем видеофайл
    video = cv2.VideoCapture(r'C:\main\lang\python\test\project\file_project\yuri_dolgoruki.mp4')

    # Проверяем, успешно ли открылся файл
    if not video.isOpened():
        print("Error opening video file")

    # Число кадров в секунду
    fps = video.get(cv2.CAP_PROP_FPS)

    # Счетчик времени
    start_time = time.time()

    while video.isOpened():
        # Читаем кадр
        ret, frame = video.read()

        if not ret:
            break

        # Сохраняем кадр каждые 3 секунды
        if video.get(cv2.CAP_PROP_POS_FRAMES) % 15 == 0:
            cv2.imwrite(r'cut_files/yuri_d/yuri_%d.jpg' % video.get(cv2.CAP_PROP_POS_FRAMES), frame)
            start_time = time.time()

        # Показываем кадр (необязательно)
        # cv2.imshow('Frame', frame)

        # Выход из цикла при нажатии клавиши 'q'
        # if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        #     break

cut_video()