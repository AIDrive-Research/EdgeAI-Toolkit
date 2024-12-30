import cv2
import numpy as np

from logger import LOGGER
from model import RknnModel


class Model(RknnModel):
    default_args = {
        'img_size': 640,
        'nms_thres': 0.45,
        'conf_thres': 0.25
    }

    def __init__(self, acc_id, name, conf):
        super().__init__(acc_id, name, conf, ['model'])

    @staticmethod
    def __preprocess_warpAffine(image, dst_width=640, dst_height=640):
        scale = min((dst_width / image.shape[1], dst_height / image.shape[0]))
        ox = (dst_width - scale * image.shape[1]) / 2
        oy = (dst_height - scale * image.shape[0]) / 2
        M = np.array([[scale, 0, ox], [0, scale, oy]], dtype=np.float32)
        img_pre = cv2.warpAffine(image, M, (dst_width, dst_height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
        IM = cv2.invertAffineTransform(M)
        img_pre = (img_pre[..., ::-1]).astype(np.float32)
        return img_pre, IM

    def __yolov8_pose_post_process(self, pred, IM=[], conf_thres=0.25, iou_thres=0.45):
        # 输入是模型推理的结果，即8400个预测框
        # 1,8400,56 [cx,cy,w,h,conf,17*3]
        boxes = []
        for img_id, box_id in zip(*np.where(pred[..., 4] > conf_thres)):
            item = pred[img_id, box_id]
            cx, cy, w, h, conf = item[:5]
            left = cx - w * 0.5
            top = cy - h * 0.5
            right = cx + w * 0.5
            bottom = cy + h * 0.5
            keypoints = item[5:].reshape(-1, 3)
            keypoints[:, 0] = keypoints[:, 0] * IM[0][0] + IM[0][2]
            keypoints[:, 1] = keypoints[:, 1] * IM[1][1] + IM[1][2]
            boxes.append([left, top, right, bottom, conf, *keypoints.reshape(-1).tolist()])
        if not boxes:
            return []
        boxes = np.array(boxes)
        lr = boxes[:, [0, 2]]
        tb = boxes[:, [1, 3]]
        boxes[:, [0, 2]] = IM[0][0] * lr + IM[0][2]
        boxes[:, [1, 3]] = IM[1][1] * tb + IM[1][2]
        keep = self._nms_boxes(boxes[:, :4], boxes[:, 4])
        nboxes = []
        if len(keep) != 0:
            nboxes = boxes[keep]
        return nboxes

    def _load_args(self, args):
        try:
            self.img_size = args.get('img_size', self.default_args['img_size'])
            self.nms_thres = args.get('nms_thres', self.default_args['nms_thres'])
            self.conf_thres = args.get('conf_thres', self.default_args['conf_thres'])
        except:
            LOGGER.exception('_load_args')
            return False
        return True

    def infer(self, image, **kwargs):
        """
        关键点检测
        Args:
            image: 图像数据，ndarray类型，RGB格式（BGR格式需转换）
        Returns: infer_result
        """
        infer_result = []
        if self.status:
            try:
                raw_width, raw_height = image.shape[1], image.shape[0]
                image, IM = self.__preprocess_warpAffine(image)
                image = np.expand_dims(image, axis=0)
                outputs = self._rknn_infer('model', [image])
                output = np.transpose(outputs[0], (0, 2, 1))
                boxes = self.__yolov8_pose_post_process(output, IM)
                if boxes is not None:
                    for box in boxes:
                        obj = {
                            'label': 0,
                            'conf': round(float(box[4]), 2),
                            'key_points': np.array(box[5:]).reshape(-1, 3).tolist()
                        }
                        xyxy = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                        obj['xyxy'] = [xyxy[0] if xyxy[0] >= 0 else 0,
                                       xyxy[1] if xyxy[1] >= 0 else 0,
                                       xyxy[2] if xyxy[2] <= raw_width else raw_width,
                                       xyxy[3] if xyxy[3] <= raw_height else raw_height]
                        infer_result.append(obj)
            except:
                LOGGER.exception('infer')
        return infer_result
