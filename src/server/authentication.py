from functools import wraps

from flask import current_app, request, abort, Response


def check_api_token_header(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_app.debug and request.headers.get('X-Api-Token') not in current_app.config['ALLOWED_API_TOKENS']:
            abort(Response('Invalid X-Api-Token', 401))
        return f(*args, **kwargs)
    return decorated_function
