import logging

import numpy as np
import torch

from image_transform import image_to_matrix
from src.models.upscaler_v2 import UpscalerV2
from PIL import Image

logger = logging.getLogger(__name__)
# might be slow for the first time
_model = torch.jit.load('model_example.pt')

# set model to eval mode
_model.eval()


checkpoint = torch.load("models/model_best_v2.ckpt", map_location=torch.device('cpu'))

learned_model = UpscalerV2()
learned_model.load_state_dict(checkpoint['model_state_dict'])

torch.save(learned_model, "models/model_best_v1.h5")


def predict(
        image,
):
    print(f'Predicting {len(image)} images')
    logger.info(f'Predicting {len(image)} images')
    # convert features to tensor
    _features = torch.tensor(image)

    # do some prediction
    with torch.no_grad():
        # prediction = _model(_features)
        results = learned_model(_features)

    # convert prediction to list
    return Image.fromarray(np.uint8(results.cpu().detach().transpose(1, 2).transpose(2, 3)[0] * 255))


if __name__ == '__main__':
    model_image = predict(torch.rand(1, 3, 32, 32).tolist())
    model_image.save('images/model_example.png')

    with open('images/image_example.png', 'rb') as f:
        inpt = image_to_matrix(f)
        print(inpt.shape)
        example_image = predict(inpt)
        example_image.save('images/model_image_example.png')
