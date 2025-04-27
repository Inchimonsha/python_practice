import serial

# Подключение к порту COMx
ser = serial.Serial('COM5', baudrate=38400)

# Отправка команды AT
ser.write(b'AT\r\n')

# Чтение ответа от устройства
response = ser.readline().decode('utf-8')
print(response)
