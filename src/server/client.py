import time
from io import BytesIO

import requests
import torch
from PIL import Image

url = 'http://127.0.0.1:5000/predict'


def response_to_image(response, path):
    if response.status_code == 200:
        # Получение байтового потока из ответа
        img_byte_array = BytesIO(response.content)

        # Открытие изображения с использованием PIL
        prediction_image = Image.open(img_byte_array)

        # Сохранение изображения
        prediction_image.save(path)

        print("Изображение успешно сохранено.")
    else:
        print(f"Ошибка запроса: {response.status_code}")
        print(response.text)


start_time = time.time()
post_response = requests.post(url, json={'images': torch.rand(1, 3, 32, 32).tolist()})
print("--- %s seconds ---" % (time.time() - start_time))

response_to_image(post_response, 'images/client_example.png')

with open('images/image_example.png', 'rb') as f:
    start_time = time.time()
    post_image_response = requests.post(url, files={'image': f})
    print("--- %s seconds ---" % (time.time() - start_time))
    response_to_image(post_image_response, 'images/client_image_example.png')
