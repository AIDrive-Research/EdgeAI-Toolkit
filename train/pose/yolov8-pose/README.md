**环境安装**


1. Clone repo and install [requirements.txt](requirements.txt) in a python>=3.8.0, including pytorch>=1.8

2. ```
   git clone https://github.com/AIDrive-Research/EdgeAI-Toolkit.git
   cd EdgeAI-Toolkit/train/pose/yolov8-pose
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
   ```

**数据准备**

   1. 数据集目录结构如下：

      ```bash
      images:
       	train
       		xxx.jpg
       	val
       		xxx.jpg
       labels:
       	train
       		xxx.txt
       	val
       		xxx.txt	
      ```

**模型训练**

```python
from ultralytics.models.yolo.pose import PoseTrainer

args = dict(model="yolov8s-pose.pt", data="coco8-pose.yaml", epochs=300)
trainer = PoseTrainer(overrides=args)
trainer.train()
```

**模型导出**

1. ONNX_RKNN导出，这里我们支持RK有NPU能力的全系列，包括RK1808、RV1109、RV1126、RK3399PRO、RK3566、RK3568、RK3588、RK3588S、RV1106、RV1103

```bash
yolo mode=export model=yolov8n-pose.pt format=onnx opset=12 simplify=True
```
