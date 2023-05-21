import numpy as np
import logging
import sys
import os
import datetime
import torch

now = datetime.datetime.now()

def get_logger(name, level=logging.INFO, filepath='./logs/log_{}.txt'.format(now.strftime("%Y-%m-%d %H-%M-%S")),
        formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(formatter)
    file_handler = logging.FileHandler(filepath, mode='w+')
    stream_handler = logging.StreamHandler(sys.stdout)

    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger

