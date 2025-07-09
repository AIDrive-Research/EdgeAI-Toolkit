import hashlib
import uuid

import shortuuid
from bson import ObjectId

from logger import LOGGER


def get_uuid1(node=None, clock_seq=None):
    return str(uuid.uuid1(node=node, clock_seq=clock_seq))


def get_uuid4():
    return str(uuid.uuid4())


def get_object_id(oid=None):
    return str(ObjectId(oid=oid))


def get_md5(*args, sep=''):
    """
    根据参数生成md5
    Args:
        *args: 参数，可以是字符串或者列表或者元组
        sep: 分隔符
    Returns:
        md5
    """
    args = [str(arg) for arg in args]
    data = sep.join(args)
    md5 = hashlib.md5(data.encode('utf-8'))
    return md5.hexdigest()


def get_short_uuid(length=None, alphabet='23456789ABCDEFGHJKLMNPQRSTUVWXYZ'):
    """
    生成短uuid
    Args:
        length: 限制长度，默认不限制
        alphabet: 字母表，用于去除易混淆字符，如数字0和字母o
    Returns: 短uuid
    """
    try:
        if alphabet is not None:
            shortuuid.set_alphabet(alphabet)
        if length is not None:
            return shortuuid.random(length=length)
        else:
            return shortuuid.uuid()
    except:
        LOGGER.exception('get_short_uuid')
    return None
