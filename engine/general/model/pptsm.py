import numpy as np
from PIL import Image

from logger import LOGGER
from model import RknnModel


class Model(RknnModel):
    default_args = {
        'img_size': 320
    }

    def __init__(self, acc_id, name, conf):
        super().__init__(acc_id, name, conf, ['model'])
        self.short_size = 340

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

    def __short_size_scale(self, image):
        h, w, _ = image.shape
        if w <= h:
            ow = self.short_size
            oh = int(self.short_size * 4.0 / 3.0)
        else:
            oh = self.short_size
            ow = int(self.short_size * 4.0 / 3.0)
        image = Image.fromarray(image, mode='RGB')
        result_image = image.resize((ow, oh), Image.BILINEAR)
        return result_image

    def __center_crop(self, image):
        w, h = image.size
        th, tw = self.img_size, self.img_size
        x1 = int(round((w - tw) / 2.0))
        y1 = int(round((h - th) / 2.0))
        result_image = image.crop((x1, y1, x1 + tw, y1 + th))
        return result_image

    def __infer_cls(self, image_list):
        trans_images = []
        for image in image_list:
            resize_image = self.__short_size_scale(image)
            crop_image = self.__center_crop(resize_image)
            trans_images.append(np.array(crop_image))
        image = np.array(trans_images)
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
                image_list = data
                output = self.__infer_cls(image_list)
                obj = {
                    'output': output.tolist()
                }
                infer_result = obj
            except:
                LOGGER.exception('infer')
        return infer_result
