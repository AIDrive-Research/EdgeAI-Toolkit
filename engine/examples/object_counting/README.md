至此，已完成快速开始中的第一步、第二步，并得到rknn模型。还需开发如下内容。

| 第三步：编写推理代码（可选）                                 | 第四步：编写后处理代码                                       | 第五步：配置文件修改&加密                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 若新算法包所需模型在本仓库提供的模型中，则无需编写推理代码。否则依照[推理模块实例](#推理模块实例)编写推理代码。注：新算法包推理代码不得与已有推理实例名称重复。 | 参照[后处理模块实例](#后处理模块实例)编写后处理代码。代码文件名称可在第三步统一修改。 | 修改算法包相关[配置文件](#算法包配置文件修改)，并对[算法包加密](#算法包加密)。 |

#### 推理模块实例

下图推理模块调用流程图。

<img src="../../../docs/assets/model_flow.png" alt="model_flow.png" style="zoom:75%;" />

主程序循环调用`__init__.py`中的`infer`函数，该函数在每个实例中会被重写。

下图为推理模块流程图。

- **模型初始化：** 加载模型，加载模型配置文件。
- **模型推理：** 读取RGB图像，进行图像缩放、维度变换等预处理，并进行模型推理。
- **结果后处理：** 执行阈值过滤与nms过滤等操作，对低阈值目标，重复目标进行过滤。
- **结果写入：** 推理结果写入redis队列。

<img src="../../../docs/assets/obj_2.png" alt="obj_2.png" style="zoom:75%;" />

以**人员计数算法**为例进行说明。算法使用`yolov5`，检测人员目标，目标检测推理代码`detect.py`如下。

```python
import cv2
import numpy as np

from logger import LOGGER
from model import RknnModel


class Model(RknnModel):
    default_args = {
        'img_size': 640,
        'nms_thres': 0.45,
        'conf_thres': 0.25,
        'anchors': [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
    }

    def __init__(self, acc_id, name, conf):
        super().__init__(acc_id, name, conf, ['model'])

    def __yolov5_post_process(self, input_data):
        masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        boxes, classes, scores = [], [], []
        for input_, mask in zip(input_data, masks):
            b, c, s = self.__process(input_, mask)
            b, c, s = self.__filter_boxes(b, c, s)
            boxes.append(b)
            classes.append(c)
            scores.append(s)
        boxes = np.concatenate(boxes)
        boxes = self._xywh2xyxy(boxes)
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)
        nboxes, nclasses, nscores = [], [], []
        keep = self._nms_boxes(boxes, scores)
        if len(keep) != 0:
            nboxes.append(boxes[keep])
            nclasses.append(classes[keep])
            nscores.append(scores[keep])
        if not nclasses and not nscores:
            return None, None, None
        return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)

    def __process(self, input_, mask):
        anchors = [self.anchors[i] for i in mask]
        grid_h, grid_w = map(int, input_.shape[0:2])
        box_confidence = input_[..., 4]
        box_confidence = np.expand_dims(box_confidence, axis=-1)
        box_class_probs = input_[..., 5:]
        box_xy = input_[..., :2] * 2 - 0.5
        col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
        col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1)
        box_xy += grid
        box_xy *= int(self.img_size / grid_h)
        box_wh = pow(input_[..., 2:4] * 2, 2)
        box_wh = box_wh * anchors
        return np.concatenate((box_xy, box_wh), axis=-1), box_confidence, box_class_probs

    def __filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter boxes with box threshold. It's a bit different with origin yolov5 post process!
        Args:
            boxes: ndarray, boxes of objects.
            box_confidences: ndarray, confidences of objects.
            box_class_probs: ndarray, class_probs of objects.
        Returns:
            boxes: ndarray, filtered boxes.
            classes: ndarray, classes for boxes.
            scores: ndarray, scores for boxes.
        """
        boxes = boxes.reshape(-1, 4)
        box_confidences = box_confidences.reshape(-1)
        box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])
        _box_pos = np.where(box_confidences >= self.conf_thres)
        boxes = boxes[_box_pos]
        box_confidences = box_confidences[_box_pos]
        box_class_probs = box_class_probs[_box_pos]
        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)
        _class_pos = np.where(class_max_score >= self.conf_thres)
        return boxes[_class_pos], classes[_class_pos], (class_max_score * box_confidences)[_class_pos]

    def _load_args(self, args):
        try:
            self.img_size = args.get('img_size', self.default_args['img_size'])
            self.nms_thres = args.get('nms_thres', self.default_args['nms_thres'])
            self.conf_thres = args.get('conf_thres', self.default_args['conf_thres'])
            self.anchors = args.get('anchors', self.default_args['anchors'])
        except:
            LOGGER.exception('_load_args')
            return False
        return True

    def infer(self, data, **kwargs):
        """
        目标检测
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
                input0_data = outputs[0].reshape([3, -1] + list(outputs[0].shape[-2:]))
                input1_data = outputs[1].reshape([3, -1] + list(outputs[1].shape[-2:]))
                input2_data = outputs[2].reshape([3, -1] + list(outputs[2].shape[-2:]))
                input_data = list()
                input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
                input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
                input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))
                boxes, classes, scores = self.__yolov5_post_process(input_data)
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
                        infer_result.append(obj)
            except:
                LOGGER.exception('infer')
        return infer_result
```

**核心函数：infer**

###### 函数输入

- `data：`RGB图像数据
- `**kwargs：`用户自定义k-v参数对

###### 函数输出

- `infer_result`，格式如下。

```python
[
    {
        "conf": 0.38,
        "label": 0,
        "xyxy": [314, 93, 435, 142]
    }, {
        "conf": 0.36,
        "label": 0,
        "xyxy": [538, 258, 553, 269]
    }
]
```

###### 处理过程

- `self._letterbox():` 对输入图像进行填充缩放。
- `self._rknn_infer():` rknn模型推理。
- `self.__yolov5_post_process():` 对rknn推理结果进行目标框、类别、置信度解码；过滤低置信度目标；非极大值抑制去除冗余目标。

#### 后处理模块实例

下图为后处理模块调用流程图。

<img src="../../../docs/assets/postprocess_flow.png" alt="postprocess_flow.png" style="zoom:75%;" />

主程序循环调用`__init__.py`中的`postprocess`函数，该函数会调用每个后处理实例中的`__process`函数。

下图为后处理模块流程图。

- **初始化：** 后处理的输入参数与输出结果初始化。
- **后处理：** 过滤非标签目标，过滤低置信度目标，目标跟踪，告警业务逻辑编写。
- **结果写入：** 结果写入告警队列。

<img src="../../../docs/assets/obj_6.png" alt="obj_6.png" style="zoom:80%;" />

`人员计数`后处理代码`person_counting.py`如下。

```python
from logger import LOGGER
from postprocessor import Postprocessor as BasePostprocessor
from tracker import Tracker


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.strategy = None
        self.tracker = None
        self.max_retain = 0
        self.targets = {}
        self.lines = None

    def __check_lost_target(self, tracker_result):
        for track_id in list(self.targets.keys()):
            if track_id not in tracker_result:
                self.targets[track_id]['lost'] += 1
            else:
                self.targets[track_id]['lost'] = 0
            if self.targets[track_id]['lost'] > self.max_retain:
                LOGGER.info('Target lost, source_id={}, alg_name={}, track_id={}, pre_target={}'.format(
                    self.source_id, self.alg_name, track_id, self.targets[track_id]['pre_target']))
                del self.targets[track_id]
        return True

    def _process(self, result, filter_result):
        hit = False
        if self.strategy is None:
            self.strategy = self.reserved_args['strategy']
        polygons = self._gen_polygons()
        if self.lines is None:
            self.lines = self._gen_lines()
        if self.tracker is None:
            self.tracker = Tracker(self.frame_interval)
            self.max_retain = self.tracker.track_buffer + 1
            LOGGER.info('Init tracker, source_id={}, alg_name={}, track_buffer={}'.format(
                self.source_id, self.alg_name, self.tracker.track_buffer))
        model_name, rectangles = next(iter(filter_result.items()))
        # 目标跟踪
        tracker_result = self.tracker.track(rectangles)
        # 检查丢失目标
        self.__check_lost_target(tracker_result)
        for track_id, rectangle in tracker_result.items():
            target = self.targets.get(track_id)
            if target is None:
                target = {
                    'lost': 0,
                    'pre_target': None
                }
                self.targets[track_id] = target
            if target['pre_target'] is not None:
                for line in self.lines.values():
                    result_ = self._cross_line_counting(
                        target['pre_target']['xyxy'], rectangle['xyxy'],
                        line['line'], line['ext']['direction'], self.strategy)
                    if result_ is not None:
                        hit = True
                        rectangle['color'] = self.alert_color
                        self._merge_cross_line_counting_result(line['ext']['result'], result_)
                        break
                else:
                    rectangle['color'] = self.non_alert_color
            else:
                rectangle['color'] = self.non_alert_color
            target['pre_target'] = rectangle
            result['data']['bbox']['rectangles'].append(rectangle)
        result['hit'] = hit
        result['data']['bbox']['polygons'].update(polygons)
        result['data']['bbox']['lines'].update(self.lines)
        return True
```

**核心函数：__process**

###### 函数输入

- `result：`结果字典。包括是否命中`hit`，以及告警结果数据`data`。若后处理产生告警，则`hit`为`true`，否则为`false`。`data`中，`rectangles`为目标框数据，`polygons`为多边形检测区域数据，`lines`为虚拟直线数据，在人员计数等算法中使用。`custom`为用户自定义的数据。

- `lines`为标记的直线`id`，`名称`，`起止点`，`颜色`与`扩展字段`。

- `direction`表示跨线的方向，支持以下几种跨线方式：`l+，r+，u+，d+，l+r-，r+l-，u+d-，d+u-`，其中，`r+`表示从左到右跨线时计数统计加1；`l+`表示从右到左跨线时计数统计加1；`u+d-`表示从下到上跨线计数统计加1，从上到下跨线时计数统计减1；依次类推。

  `action`定义了跨线行为的字面描述，当`direction`为`l+`，`r+`，`u+`，`d+`中的一种时，可选属性为`count`，表示计数统计结果；当`direction`为`l+r-`，`r+l-`，`u+d-`，`d+u-`中的一种时，可选属性为`increase，decrease，delta`。`increase`的值会在触发加1行为时加1，`decrease`的值会在触发减1行为时加1，`delta`的值为`increase - decrease`，表示最终的净增值。

```json
{
	"hit": true,
	"data": {
		"bbox": {
			"rectangles": [{
				"xyxy": [474, 206, 618, 568],
				"color": [0, 0, 255],
				"label": "人",
				"conf": 0.9,
				"ext": {}
			}],
			"polygons": {},
			"lines": {
				"line_7673b63d-45b6-4d59-8f7b-9f81908932f5": {
					"name": "进入人数",
					"line": [
						[470, 280],
						[735, 710]
					],
					"color": [255, 0, 0]
				},
				"ext": {
					"direction": "r+",
					"action": {
						"count": "统计"
					},
					"result": {
						"count": 3,
						"increase": 0,
						"decrease": 0,
						"delta": 0
					}
				}
			}
		}
	},
	"custom": {}
}
```

- `filter_result`：标签过滤以及低置信度过滤后的结果。`person`指的是模型名称，后面的值是模型的推理结果list。

```json
{
	"person": [{
		"xyxy": [474, 206, 618, 568],
		"color": [0, 0, 255],
		"label": "人",
		"conf": 0.9,
		"ext": {}
	}]
}
```

###### 函数输出

`True` or `False`

#### 算法包配置文件修改

至此，已得到模型文件，推理代码，后处理代码。下面对算法名称、模型名称、yaml配置文件、json文件修改。

| 修改项             | 详情                                                         |
| ------------------ | ------------------------------------------------------------ |
| 算法名称           | 算法包文件夹名称、后处理代码文件名称、后处理json文件名称以及postprocessor.yaml中的name需要保持一致。新算法名称不得与已有算法名称重复。 |
| 模型名称           | 模型文件夹名称、model.yaml中的模型名称、postprocessor.yaml中的模型名称、xxx.json中的模型名称需保持一致。新模型名称不得与已有模型名称重复。 |
| model.yaml         | 修改模型名称、检测类型（模型名称）、输入参数、推理时间。     |
| postprocessor.yaml | 修改算法中英文名称、算法描述、分组类别、模型配置。           |
| xxx.json           | 修改json文件中的模型名称、算法名称、语音文本等内容。         |
| 模型文件           | 模型文件夹下的模型，统一命名model。                          |

- 算法名称修改为自己定义的算法名称，如：custom_person_intrusion，若不修改，会覆盖已有算法文件

![](../../../docs/assets/algname.png)

- 模型名称修改为自定义的算法名称，如：custom_person，若不修改，会覆盖已有模型。将量化好的模型放置下图第一个文件夹下

![](../../../docs/assets/modelname.png)

- 推理代码文件名称修改：在`区域入侵`算法中，推理代码文件为`detect.py`，将此文件名称修改为`custom_detect.py`。

- `model.yaml`文件修改：`model.yaml`中的①为模型名称，即模型文件夹名称；②为检测类型，与推理代码文件名称一致；③为推理时间。

![](../../../docs/assets/obj_8.png)

- `postprocessor.yaml`文件修改：①处为算法名称，②处为算法中文名，③为算法分组名称，④为模型名称，⑤为label映射，⑥为告警label，矩形框上的告警label，可自行自定义，若不写，使用label_map中的value

![](../../../docs/assets/obj_9.png)

- 前端配置文件xxx.json修改：详细可参照[前端配置文件](../../JSON.md)。修改`model_args`中的模型文件名。

![](../../../docs/assets/obj_10.png)

#### 算法包加密

- 将算法包加密为bin文件。下载算法包[加密工具](https://pan.baidu.com/s/173r6sLMh77n3JrKkLnSpYg?pwd=0000)。将待加密算法包放在文件夹内（文件夹只含单个算法包），填写待加密算法包的上级路径（如下述文件夹所示），点击【确定】按钮，提示即将加密的算法包名称，点击【ok】；

![](../../../docs/assets/encryption_1.png)

![](../../../docs/assets/encryption_2.png)

- 加密完成的bin文件为最终文件，从盒子后台管理系统【算法仓库】中导入即可。