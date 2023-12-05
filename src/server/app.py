import logging
import os
from io import BytesIO

from flask import request, Flask, jsonify, Response, send_from_directory

import setup_logging
from authentication import check_api_token_header
from image_transform import image_to_matrix
from model import predict
import dotenv

setup_logging.setup()
dotenv.load_dotenv()
app = Flask(__name__)
app.config['ALLOWED_API_TOKENS'] = os.getenv('API_TOKENS').split(',')

logger = logging.getLogger(__name__)


@app.route('/ping')
def ping():
    logger.info('ping received')
    return 'pong'


@app.route('/')
@check_api_token_header
def index():
    return send_from_directory('static', 'index.html')


@app.route('/predict', methods=['POST'])
@check_api_token_header
def post_predict_handler() -> Response:
    if request.mimetype == 'application/json':
        features = request.json['images']
    else:
        file = request.files['image']
        features = image_to_matrix(file)
    prediction = predict(features)

    img_byte_array = BytesIO()
    prediction.save(img_byte_array, format='PNG')
    img_byte_array = img_byte_array.getvalue()

    response = Response(img_byte_array, content_type='image/png')
    response.headers['Content-Disposition'] = 'inline; filename=prediction.png'

    return response


if __name__ == '__main__':
    app.run()
