import numpy as np

from logger import LOGGER
from model import RknnModel


class Model(RknnModel):
    default_args = {
        'img_size': 224,
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

    @staticmethod
    def __softmax(x):
        # 计算指数
        exp_x = np.exp(x)
        # 对每行求和
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        # 计算softmax
        softmax_output = exp_x / sum_exp_x
        return softmax_output

    def __infer_cls(self, image):
        image, _, _ = self._letterbox(image, (self.img_size, self.img_size), stretch=True)
        image = np.expand_dims(image, axis=0)
        outputs = self._rknn_infer('model', [image])
        return self.__softmax(outputs[0]).reshape(-1)

    def infer(self, data, **kwargs):
        """
        图像分类
        Args:
            data: 图像数据，ndarray类型，RGB格式（BGR格式需转换）
        Returns: infer_result
        """
        infer_result = None
        if self.status:
            try:
                image = data
                output = self.__infer_cls(image)
                obj = {
                    'output': output.tolist()
                }
                infer_result = obj
            except:
                LOGGER.exception('infer')
        return infer_result
