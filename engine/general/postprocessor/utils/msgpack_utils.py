import msgpack

from logger import LOGGER


def load(data, raw=False):
    try:
        return msgpack.unpackb(data, raw=raw)
    except:
        LOGGER.exception('load')
    return None


def dump(data, use_bin_type=True):
    try:
        return msgpack.packb(data, use_bin_type=use_bin_type)
    except:
        LOGGER.exception('dump')
    return None
