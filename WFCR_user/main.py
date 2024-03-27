import requests
import glob
import os
import time

from WFCR_user.wfcr import WFCR
from qreader import QReader
import cv2

reader = WFCR()
qreader = QReader()


def get_prediction(input_image):
    img, img_bbox, img_mask = reader.detect(input_image,
                                            threshold=0.8)  # отправка изображений в детектор (можно изменить точность детекции параметром threshold(по умолчанию 80%))
    result, indx = reader.recognize_values(img_bbox)
    qrCode = qreader.detect_and_decode(image=cv2.cvtColor(cv2.imread(input_image), cv2.COLOR_BGR2RGB))

    if indx > 1:
        img_bbox = img_bbox.rotate(180)

    return img_bbox, result, qrCode


def get_value(input_image):
    img, img_bbox, img_mask = reader.detect(input_image,
                                            threshold=0.8)  # отправка изображений в детектор (можно изменить точность детекции параметром threshold(по умолчанию 80%))
    qrCode = qreader.detect_and_decode(image=cv2.cvtColor(cv2.imread(input_image), cv2.COLOR_BGR2RGB))
    result = reader.recognize_values(img_bbox)[0]
    return result, qrCode


def create_api_data_image(image_url):
    result, qrCode = get_value(image_url)
    qr = ""
    for i in qrCode:
        qr += str(i)
    url = "http://127.0.0.1:8000/api/data/"
    data = {
        "meter": f"{result}",
        "qr": f"{qr}"
    }
    response = requests.post(url, json=data)


# create_api_data_image
#
# folder_path = 'images/'
#
# while True:
#     for file_path in glob.glob('images/*.jpg') + glob.glob('images/*.png'):
#         # Проверяем, что файл является изображением
#         if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
#             try:
#                 create_api_data_image(file_path)
#                 os.remove(file_path)
#             except Exception as e:
#                 print(f"Ошибка при удалении или обработке файла {file_path}: {e}")
#         else:
#             print(f"Неправильное расширение файла: {file_path}")
#
#     time.sleep(60)
