import json

from logger import LOGGER


def loads(data):
    try:
        return json.loads(data)
    except:
        LOGGER.exception('loads')
    return {}


def load(path, mode='r', encoding='utf-8'):
    try:
        with open(path, mode=mode, encoding=encoding) as f:
            return json.load(f)
    except:
        LOGGER.exception('load')
    return {}


def dumps(data, ensure_ascii=False):
    try:
        return json.dumps(data, ensure_ascii=ensure_ascii)
    except:
        LOGGER.exception('dumps')
    return None


def dump(data, path, mode='w', encoding='utf-8', ensure_ascii=False):
    try:
        with open(path, mode=mode, encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=ensure_ascii)
            return True
    except:
        LOGGER.exception('dump')
    return False
