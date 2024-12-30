import cv2
import numpy as np
from skimage import transform

from logger import LOGGER
from model import RknnModel


class Model(RknnModel):
    default_args = {
        'max_num': 5,
        # detection
        'img_size': 640,
        'nms_thres': 0.4,
        'conf_thres': 0.5,
        'fmc': 3,
        'feat_stride_fpn': [8, 16, 32],
        'num_anchors': 2,
        'center_cache': {},
        'use_kps': True,
        # alignment
        'reference_facial_points': [
            [30.29459953, 51.69630051],
            [65.53179932, 51.50139999],
            [48.02519989, 71.73660278],
            [33.54930115, 87],
            [62.72990036, 87]
        ],
        'crop_size': (96, 112),
        'face_size': (112, 112)
    }

    def __init__(self, acc_id, name, conf):
        super().__init__(acc_id, name, conf, ['det', 'rec', 'quality'])
        self.ref_pts = self.__get_reference_facial_points(output_size=self.face_size)
        self.transform = transform.SimilarityTransform()

    @staticmethod
    def __expand_crop(image_shape, rect, expand_ratio=0.3):
        imgh, imgw, _ = image_shape
        xmin, ymin, xmax, ymax = rect[0], rect[1], rect[2], rect[3]
        h_half = (ymax - ymin) * (1 + expand_ratio) / 2.
        w_half = (xmax - xmin) * (1 + expand_ratio) / 2.
        if h_half > w_half * 4 / 3:
            w_half = h_half * 0.75
        center = [(ymin + ymax) / 2., (xmin + xmax) / 2.]
        ymin = max(0, int(center[0] - h_half))
        ymax = min(imgh - 1, int(center[0] + h_half))
        xmin = max(0, int(center[1] - w_half))
        xmax = min(imgw - 1, int(center[1] + w_half))
        return [xmin, ymin, xmax, ymax]

    @staticmethod
    def __distance2bbox(points, distance, dw, dh, max_shape=None):
        x1 = points[:, 0] - distance[:, 0] - dw
        y1 = points[:, 1] - distance[:, 1] - dh
        x2 = points[:, 0] + distance[:, 2] - dw
        y2 = points[:, 1] + distance[:, 3] - dh
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def __distance2kps(points, distance, dw, dh, max_shape=None):
        distance = distance.reshape(distance.shape[0], -1, 2)
        px = points[:, None, :2] + distance
        px[:, :, 0] -= dw
        px[:, :, 1] -= dh
        if max_shape is not None:
            px[:, :, 0] = np.clip(px[:, :, 0], 0, max_shape[1])
            px[:, :, 1] = np.clip(px[:, :, 1], 0, max_shape[0])
        return px.reshape(points.shape[0], -1)

    def __post_process(self, outputs, dw, dh, image_shape):
        score_list = []
        bbox_list = []
        kps_list = []
        image_shape = image_shape[1:]
        for idx, stride in enumerate(self.feat_stride_fpn):
            scores = outputs[idx][0, :, :]
            bbox_preds = outputs[idx + self.fmc][0, :, :] * stride
            if self.use_kps:
                kps_preds = outputs[idx + self.fmc * 2][0, :, :] * stride
            height = self.img_size // stride
            width = self.img_size // stride
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if self.num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * self.num_anchors, axis=1).reshape((-1, 2))
            pos_inds = np.where(scores >= self.conf_thres)[0]
            bboxes = self.__distance2bbox(anchor_centers, bbox_preds, dw, dh, image_shape)
            for i in range(len(bboxes)):
                bboxes[i] = self.__expand_crop(image_shape, bboxes[i], expand_ratio=0.1)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            if len(pos_scores):
                score_list.append(pos_scores)
                bbox_list.append(pos_bboxes)
                if self.use_kps:
                    kpss = self.__distance2kps(anchor_centers, kps_preds, dw, dh, image_shape)
                    kpss = kpss.reshape((kpss.shape[0], -1, 2))
                    pos_kpss = kpss[pos_inds]
                    kps_list.append(pos_kpss)
        return bbox_list, score_list, kps_list

    def __infer_det(self, image):
        # 预处理
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
        outputs = self._rknn_infer('det', [image])
        # 后处理
        bbox_list, score_list, kps_list = self.__post_process(outputs, dw, dh, image.shape)
        if not bbox_list:
            return None
        # 合并 score_list、bbox_list、kps_list
        scores = np.vstack(score_list)
        bboxes = np.vstack(bbox_list) / scale
        kpss = np.vstack(kps_list) / scale if self.use_kps else None
        # 计算检测结果
        order = scores.ravel().argsort()[::-1]
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order]
        keep = self._nms_boxes(pre_det[:, :4], pre_det[:, -1])
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order][keep, :, :]
        # 限制人脸个数
        if self.max_num > 0 and det.shape[0] > self.max_num:
            det = det[:self.max_num]
            kpss = kpss[:self.max_num]
        return det[:, :4], det[:, -1], kpss

    def __get_reference_facial_points(self, output_size=(112, 112)):
        reference_facial_points = np.array(self.reference_facial_points)
        crop_size = np.array(self.crop_size)
        x_scale = output_size[0] / crop_size[0]
        y_scale = output_size[1] / crop_size[1]
        reference_facial_points[:, 0] *= x_scale
        reference_facial_points[:, 1] *= y_scale
        return reference_facial_points

    def __face_alignment(self, image, boxes, landmarks):
        face_warped = []
        for i, src_pts in enumerate(landmarks):
            box = boxes[i]
            src_pts[:, 0] -= box[0]
            src_pts[:, 1] -= box[1]
            box = box.astype(np.int32)
            if max(src_pts.shape) < 3 or min(src_pts.shape) != 2:
                LOGGER.error('facial_pts.shape must be (K,2) or (2,K) and K>2')
            if src_pts.shape[0] == 2:
                src_pts = src_pts.T
            if src_pts.shape != self.ref_pts.shape:
                LOGGER.error('facial_pts and reference_pts must have the same shape')
            self.transform.estimate(src_pts, self.ref_pts)
            face_img = cv2.warpAffine(
                image[box[1]: box[3], box[0]:box[2]], self.transform.params[0:2, :], self.face_size)
            face_warped.append(face_img)
        return face_warped

    def __extract_feature(self, face_warped):
        features = []
        for image in face_warped:
            image = np.expand_dims(image, axis=0)
            feature = self._rknn_infer('rec', [image])
            norm = np.linalg.norm(feature[0], ord=2, axis=1, keepdims=True)
            feature = feature / norm
            feature = feature.reshape(-1)
            features.append(feature)
        return features

    def __infer_quality(self, face_warped):
        face_quality = []
        for image in face_warped:
            image = np.expand_dims(image, axis=0)
            quality = self._rknn_infer('quality', [image])[0][0][0]
            face_quality.append(quality)
        return face_quality

    def _load_args(self, args):
        try:
            self.max_num = args.get('max_num', self.default_args['max_num'])
            self.img_size = args.get('img_size', self.default_args['img_size'])
            self.nms_thres = args.get('nms_thres', self.default_args['nms_thres'])
            self.conf_thres = args.get('conf_thres', self.default_args['conf_thres'])
            self.fmc = args.get('fmc', self.default_args['fmc'])
            self.feat_stride_fpn = args.get('feat_stride_fpn', self.default_args['feat_stride_fpn'])
            self.num_anchors = args.get('num_anchors', self.default_args['num_anchors'])
            self.center_cache = args.get('center_cache', self.default_args['center_cache'])
            self.use_kps = args.get('use_kps', self.default_args['use_kps'])
            self.reference_facial_points = args.get(
                'reference_facial_points', self.default_args['reference_facial_points'])
            self.crop_size = args.get('crop_size', self.default_args['crop_size'])
            self.face_size = args.get('face_size', self.default_args['face_size'])
        except:
            LOGGER.exception('_load_args')
            return False
        return True

    def infer(self, data, **kwargs):
        """
        目标检测+人脸特征提取
        Args:
            data: 图像数据，ndarray类型，RGB格式（BGR格式需转换）
        Returns: infer_result
        """
        infer_result = []
        if self.status:
            try:
                image = data
                ret = self.__infer_det(image)
                if ret is not None:
                    boxes, scores, landmarks = ret
                    face_warped = self.__face_alignment(image, boxes, landmarks)
                    face_quality = self.__infer_quality(face_warped)
                    face_features = self.__extract_feature(face_warped)
                    for zip_ in zip(boxes, scores, face_quality, face_features):
                        obj = {
                            'conf': round(float(zip_[1]), 2),
                            'quality': round(float(zip_[2]), 2),
                            'feature': zip_[3].tolist()
                        }
                        xyxy = zip_[0]
                        xyxy = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                        obj['xyxy'] = [xyxy[0] if xyxy[0] >= 0 else 0,
                                       xyxy[1] if xyxy[1] >= 0 else 0,
                                       xyxy[2] if xyxy[2] <= image.shape[1] else image.shape[1],
                                       xyxy[3] if xyxy[3] <= image.shape[0] else image.shape[1]]
                        infer_result.append(obj)
            except:
                LOGGER.exception('infer')
        return infer_result
