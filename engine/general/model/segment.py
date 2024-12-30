import time

import cv2
import numpy as np

from logger import LOGGER
from model import RknnModel
from utils.image_utils import opencv_to_base64


class Model(RknnModel):
    default_args = {
        'max_wh': 7680,  # maximum box width and height
        'max_nms': 30000,  # maximum number of boxes
        'img_size': 640,
        'nms_thres': 0.45,
        'conf_thres': 0.25
    }

    def __init__(self, acc_id, name, conf):
        super().__init__(acc_id, name, conf, ['model'])

    def __yolov5_seg_post_process(self, pred, proto):
        pred = self.__process_det(pred, max_det=1000, nm=32)
        nboxes, nscores, nclasses = pred[:, :4], pred[:, 4], pred[:, 5]
        nmasks = self.__process_mask(proto[0], pred[:, 6:], nboxes, (self.img_size, self.img_size), upsample=True)
        nmasks = (nmasks * 255).astype(np.uint8)
        return nboxes, nclasses, nscores, nmasks

    def __process_det(self, prediction, classes=None, agnostic=False, multi_label=False, labels=(), max_det=300, nm=0):
        if isinstance(
                prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output
        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - nm - 5  # number of classes
        xc = prediction[..., 4] > self.conf_thres  # candidates
        # Checks
        assert 0 <= self.conf_thres <= 1, 'Invalid Confidence threshold {}, valid values are between 0.0 and 1.0'.format(
            self.conf_thres)
        assert 0 <= self.nms_thres <= 1, 'Invalid IoU {}, valid values are between 0.0 and 1.0'.format(self.nms_thres)
        # Settings
        time_limit = 0.5 + 0.05 * bs  # seconds to quit after
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        t = time.time()
        mi = 5 + nc  # mask start index
        output = [np.zeros((0, 6 + nm))] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            x = x[xc[xi]]  # confidence
            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = np.zeros((len(lb), nc + nm + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
                x = np.concatenate((x, v), 0)
            # If none remain process next image
            if not x.shape[0]:
                continue
            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            # Box/Mask
            box = self._xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
            mask = x[:, mi:]  # zero columns if no masks
            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:mi] > self.conf_thres).nonzero(as_tuple=False).T
                x = np.concatenate((box[i], x[i, 5 + j, None], j[:, None].astype(np.float32), mask[i]), 1)
            else:
                # conf 包含最大值
                conf = x[:, 5:mi].max(axis=1, keepdims=True)
                # j 包含最大值的索引
                j = np.argmax(x[:, 5:mi], axis=1, keepdims=True)
                x = np.concatenate((box, conf, j.astype(np.float32), mask), 1)[conf.reshape(-1) > self.conf_thres]
            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == np.array(classes)).any(1)]
            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > self.max_nms:  # excess boxes
                x = x[np.argsort(x[:, 4])[::-1]][:self.max_nms]
            else:
                x = x[np.argsort(x[:, 4])[::-1]]
            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else self.max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = self._nms_boxes(boxes, scores)  # NMS
            if i.shape[0] > max_det:
                i = i[:max_det]
            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                LOGGER.warning('WARNING NMS time limit {:.3f}s exceeded'.format(time_limit))
                break
        return output[0]

    def __process_mask(self, protos, masks_in, bboxes, shape, upsample=False):
        """
        Crop before upsample.
        proto_out: [mask_dim, mask_h, mask_w]
        out_masks: [n, mask_dim], n is number of masks after nms
        bboxes: [n, 4], n is number of masks after nms
        shape:input_image_size, (h, w)

        return: h, w, n
        """
        c, mh, mw = protos.shape  # CHW
        ih, iw = shape
        masks = self.__sigmoid(masks_in @ protos.astype(np.float32).reshape(c, -1)).reshape(-1, mh, mw)  # CHW
        downsampled_bboxes = bboxes.copy()
        downsampled_bboxes[:, 0] *= mw / iw
        downsampled_bboxes[:, 2] *= mw / iw
        downsampled_bboxes[:, 3] *= mh / ih
        downsampled_bboxes[:, 1] *= mh / ih
        masks = self.__crop_mask(masks, downsampled_bboxes)  # CHW
        if upsample:
            target_shape = shape[:2]
            resampled = np.zeros((masks.shape[0],) + target_shape)
            for i in range(masks.shape[0]):
                resampled[i] = cv2.resize(masks[i], (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
            masks = resampled
        return np.where(masks > 0.5, 1, 0)

    @staticmethod
    def __sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def __crop_mask(masks, boxes):
        """
        "Crop" predicted masks by zeroing out everything not in the predicted bbox.
        Vectorized by Chong (thanks Chong).

        Args:
            - masks should be a size [h, w, n] tensor of masks
            - boxes should be a size [n, 4] tensor of bbox coords in relative point form
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
        c = np.arange(h, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def _load_args(self, args):
        try:
            self.max_wh = args.get('max_wh', self.default_args['max_wh'])
            self.max_nms = args.get('max_nms', self.default_args['max_nms'])
            self.img_size = args.get('img_size', self.default_args['img_size'])
            self.nms_thres = args.get('nms_thres', self.default_args['nms_thres'])
            self.conf_thres = args.get('conf_thres', self.default_args['conf_thres'])
        except:
            LOGGER.exception('_load_args')
            return False
        return True

    def infer(self, data, **kwargs):
        """
        实例分割
        Args:
            data: 图像数据，ndarray类型，RGB格式（BGR格式需转换）
        Returns: infer_result
        """
        infer_result = []
        if self.status:
            try:
                image = data
                scale = 1
                raw_width, raw_height = image.shape[1], image.shape[0]
                if max(image.shape[:2]) != self.img_size:
                    scale = self.img_size / max(image.shape[:2])
                    if raw_height > raw_width:
                        image = cv2.resize(image, (int(raw_width * scale), self.img_size))
                    else:
                        image = cv2.resize(image, (self.img_size, int(raw_height * scale)))
                image, dw, dh = self._letterbox(image, (self.img_size, self.img_size))
                image = np.expand_dims(image, axis=0)
                outputs = self._rknn_infer('model', [image])
                pred = outputs[0]
                proto = outputs[1]
                boxes, classes, scores, masks = self.__yolov5_seg_post_process(pred, proto)
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        obj = {
                            'label': int(classes[i]),
                            'conf': round(float(scores[i]), 2)
                        }
                        xyxy = [int(box[0] - dw), int(box[1] - dh), int(box[2] - dw), int(box[3] - dh)]
                        if scale != 1:
                            xyxy = [int(x / scale) for x in xyxy]
                        obj['xyxy'] = [xyxy[0] if xyxy[0] >= 0 else 0,
                                       xyxy[1] if xyxy[1] >= 0 else 0,
                                       xyxy[2] if xyxy[2] <= raw_width else raw_width,
                                       xyxy[3] if xyxy[3] <= raw_height else raw_height]
                        mask_shape = masks[i].shape
                        obj['mask'] = opencv_to_base64(
                            masks[i][int(dh):int(mask_shape[0] - dh), int(dw):int(mask_shape[1] - dw)])
                        infer_result.append(obj)
            except:
                LOGGER.exception('infer')
        return infer_result
