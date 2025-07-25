from logger import LOGGER
from utils.image_utils.turbojpegutils import mat_to_bytes


class Model:
    def __init__(self, acc_id, name, conf):
        self.acc_id = acc_id
        self.name = name
        self.conf = conf

    def infer(self, data, **kwargs):
        """
        不做推理，直接返回原始数据
        Args:
            data: 推理数据
        Returns: infer_result
        """
        try:
            return mat_to_bytes(data)
        except:
            LOGGER.exception('infer')
        return None
