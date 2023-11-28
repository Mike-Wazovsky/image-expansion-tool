from io import BytesIO

import numpy as np
import torch
from PIL import Image


def image_to_matrix(file):
    # Load the image from bytes
    img = Image.open(file)
    a = torch.tensor(np.array(img, dtype=np.float32)).transpose(1, 2).transpose(0, 1)
    b = a.numpy()
    c = torch.tensor([b[0:3]])
    return c
    # TODO(fix image to matrix)


if __name__ == '__main__':
    with open('images/image_example.png', 'rb') as f:
        res = image_to_matrix(f)
        print(type(res))
        new_image = Image.fromarray(np.uint8(res.transpose(0, 1).transpose(1, 2)))
        new_image.save('images/image_to_matrix_test.png')

