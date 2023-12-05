import logging
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'DEBUG',
        },
        # You can uncomment this to log to a file, don't forget to add the file_handler to the loggers[*]['handlers']
        # 'file_handler': {
        #     'class': 'logging.handlers.RotatingFileHandler',
        #     'filename': 'myapp.log',
        #     'formatter': 'standard',
        #     'level': 'DEBUG',
        #     'maxBytes': 10485760,  # 10MB
        #     'backupCount': 3,      # Keep up to 3 backup logs
        # },
    },
    'loggers': {
        'waitress': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
        'root': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
    }
}


def setup():
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger('logger')
    logger.info('Logging configured')
