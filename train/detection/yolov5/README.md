**环境安装**

1. Clone repo and install [requirements.txt](requirements.txt) in a python=3.8.0 environment, including pytorch>=1.8

2. ```
   git clone https://github.com/AIDrive-Research/EdgeAI-Toolkit.git
   cd EdgeAI-Toolkit/train/detection/yolov5
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
   ```

**数据准备**

1. 制作VOC格式数据集，目录结构如下。

   ```
    JPEGImages:
    	xxx.jpg
    Annotations:
    	xxx.xml
   ```

2. 将VOC格式数据集转换成YOLO格式

   修改tools/0_xml2txt.py：

   - CLASSES：目标类别
   - XML_DIR：VOC标注文件路径
   - JPEG_DIR：图片文件路径
   - LABEL_DIR：YOLO格式标注文件存储路径

   ```
    python tools/0_xml2txt.py
   ```

3. 将YOLO格式数据拆分训练集和验证集

   ```
    python tools/1_split_train_val.py --input-images 图片文件路径 --input-labels YOLO格式标注文件存储路径 --output YOLO数据集存储路径
   ```

   目录结构如下图:

   ```
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

4. 新建训练数据yaml文件，参考data/coco128.yaml编辑自己的yaml文件

   ```
    cd data
    cp coco128.yaml custom.yaml
   ```

   修改custom.yaml:

   - path: 3中YOLO数据集存储路径
   - train: images/train
   - val: image/val
   - nc: 类别数目
   - names: 类别名称，此处与2中CLASSES相同

**训练**

下载预训练权重：[./yolov5s.pt](https://pan.baidu.com/s/1eGCl5q809TVYe8vh7heh3A?pwd=0000)

1. 单卡训练

   ```
    python train.py --data data/custom.yaml --epochs 300 --weights '' --cfg models/yolov5s.yaml --batch-size 128 --device 0
   ```

   如果使用预训练

   ```
    python train.py --data data/custom.yaml --epochs 300 --weights yolov5s.pt --batch-size 128 --device 0
   ```

2. 推荐使用多卡训练

   ```
    python -m torch.distributed.run --nproc_per_node 2 train.py --batch 64 --data data/custom.yaml --weights yolov5s.pt --device 0,1
   ```

**模型导出**

   1. ONNX_RKNN导出，这里我们支持RK有NPU能力的全系列，包括RK1808、RV1109、RV1126、RK3399PRO、RK3566、RK3568、RK3588、RK3588S、RV1106、RV1103

      ```
      python export_rk.py --weights xxx.pt --include onnx --simplify --opset 12 --rknpu rk3588
      ```

- ks968产品，rknpu为rk3588；ks916产品rknpu为rk3568