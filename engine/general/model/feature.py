import numpy as np

from logger import LOGGER
from model import RknnModel


class Model(RknnModel):
    default_args = {
        'img_size': 224
    }

    def __init__(self, acc_id, name, conf):
        super().__init__(acc_id, name, conf, ['model'])

    def _load_args(self, args):
        try:
            self.img_size = args.get('img_size', self.default_args['img_size'])
        except:
            LOGGER.exception('_load_args')
            return False
        return True

    def __get_feature(self, image):
        image, _, _ = self._letterbox(image, (self.img_size, self.img_size), stretch=True)
        image = np.expand_dims(image, axis=0)
        outputs = self._rknn_infer('model', [image])
        norm = np.linalg.norm(outputs[0], ord=2, axis=1, keepdims=True)
        feature = outputs[0] / norm
        feature = feature.reshape(-1)
        return np.array(feature, dtype=np.float32)

    def infer(self, data, **kwargs):
        """
        特征提取
        Args:
            data: 图像数据，ndarray类型，RGB格式（BGR格式需转换）
        Returns: infer_result
        """
        infer_result = None
        if self.status:
            try:
                image = data
                feature = self.__get_feature(image)
                infer_result = feature.tolist()
            except:
                LOGGER.exception('infer')
        return infer_result
