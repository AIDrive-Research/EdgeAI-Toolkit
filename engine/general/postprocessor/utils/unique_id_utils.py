import uuid

from bson import ObjectId


def get_uuid1(node=None, clock_seq=None):
    return str(uuid.uuid1(node=node, clock_seq=clock_seq))


def get_uuid4():
    return str(uuid.uuid4())


def get_object_id(oid=None):
    return str(ObjectId(oid=oid))
