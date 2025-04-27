import os

# Задаем директорию с файлами
directory = r'C:\main\lang\python\test\project\file_project\f_xml'

# Получаем список файлов в директории
files = os.listdir(directory)

# Сортируем файлы в порядке их следования
files.sort()

# Переименовываем файлы
for i, filename in enumerate(files, start=1):
    old_path = os.path.join(directory, filename)
    new_name = f"{i:04d}{os.path.splitext(filename)[1]}"
    new_path = os.path.join(directory, new_name)
    os.rename(old_path, new_path)
    print(f"Переименован файл: {filename} -> {new_name}")