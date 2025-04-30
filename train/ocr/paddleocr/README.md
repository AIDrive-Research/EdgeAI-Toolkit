## 环境安装
1. Clone repo and install [requirements.txt](https://github.com/AIDrive-Research/EdgeAI-Toolkit/blob/main/train/ocr/paddleocr/requirements.txt) in a python=3.8.0 environment, including pytorch>=1.8 推荐使用Conda虚拟环境。

2. ```bash
    git clone https://github.com/AIDrive-Research/EdgeAI-Toolkit.git
    cd EdgeAI-Toolkit/train/ocr/paddleocr
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
   ```

## 数据准备
1. 标注数据集，结构如下：
   ```bash
   |-train_data
      |-crop_img
         |- word_001_crop_0.png
         |- word_002_crop_0.jpg
         |- word_003_crop_0.jpg
         | ...
      | Label.txt
      | rec_gt.txt
      |- word_001.png
      |- word_002.jpg
      |- word_003.jpg
      | ...
   ```

2. 在终端中输入以下命令执行数据集划分脚本：

   ```bash
   cd ./PPOCRLabel 
   python gen_ocr_train_val_test.py --trainValTestRatio 6:2:2 --datasetRootPath ../train_data
   ```
   参数说明：

   - trainValTestRatio 是训练集、验证集、测试集的图像数量划分比例，根据实际情况设定，默认是6:2:2
   - datasetRootPath 是PPOCRLabel标注的完整数据集存放路径

3. 在当前路径生成训练集和验证集
   目录结构如下图:

   ```bash
   |-det:
      |-test
         |- xxx.jpg
      |-train
         |- xxx.jpg
      |-val
         |- xxx.jpg
      | test.txt
      | train.txt
      | val.txt
    rec:
      |-test
         |- xxx.jpg
      |-train
         |- xxx.jpg	
      |-val
         |- xxx.jpg
      | test.txt
      | train.txt
      | val.txt   
   ```

## 预训练模型下载

* 参考：[https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/ppocr/model\_list.md](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/ppocr/model_list.md)
* 模型：ch\_PP-OCRv4\_det、ch\_PP-OCRv4\_rec

## 训练超参选择
在模型微调的时候，最重要的超参就是预训练模型路径pretrained_model, 学习率learning_rate与batch_size，部分配置文件如下所示。
```bash
Global:
  pretrained_model: ./configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_student.yml # 预训练模型路径
Optimizer:
  lr:
    name: Cosine
    learning_rate: 0.001 # 学习率
    warmup_epoch: 2
  regularizer:
    name: 'L2'
    factor: 0

Train:
  loader:
    shuffle: True
    drop_last: False
    batch_size_per_card: 8  # 单卡batch size
    num_workers: 4
```
上述配置文件中，首先需要将pretrained_model字段指定为student.pdparams文件路径。


## 训练

Det训练：

* 参考：[https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/ppocr/model\_train/detection.md](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/ppocr/model_train/detection.md)

```Plain Text
python3 -m paddle.distributed.launch --gpus '2' tools/train.py -c configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_student.yml -o Global.pretrained_model=./pretrained_model/ch_PP-OCRv4_det_train/best_accuracy.pdparams
```

Rec训练：

* 参考：[https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/ppocr/model\_train/recognition.md](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/ppocr/model_train/recognition.md)

```bash
python3 -m paddle.distributed.launch --gpus '3'  tools/train.py -c ./configs/rec/PP-OCRv4/ch_PP-OCRv4_rec.yml -o Global.pretrained_model=./pretrained_model/ch_PP-OCRv4_rec_train/student.pdparams
```


## 导出ONNX

检测模型转inference 模型方式：

```bash
python3 tools/export_model.py -c configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_student.yml -o Global.pretrained_model="./output/yibiao_ocr/det/best_model/model" Global.save_inference_dir="./output/yibiao_ocr/det/det_inference/"
```

DB检测模型inference 模型预测：

```bash
python3 tools/infer/predict_det.py --det_algorithm="DB" --det_model_dir="./output/yibiao_ocr/det/det_inference/" --image_dir="./images/" --use_gpu=True
```

Det模型转onnx：

```Plain Text
paddle2onnx --model_dir ./output/yibiao_ocr/det/det_inference --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ./output/yibiao_ocr/det/det_inference/yibiao_det.onnx --opset_version 12 --enable_dev_version True --enable_onnx_checker True
```
```Plain Text
python -m paddle2onnx.optimize --input_model ./output/yibiao_ocr/det/det_inference/yibiao_det.onnx --output_model ./output/yibiao_ocr/det/det_inference/yibiao_det_op.onnx --input_shape_dict "{'x':[1,3,640,640]}"
```


识别模型转inference模型方式：

```bash
python3 tools/export_model.py -c ./configs/rec/PP-OCRv4/ch_PP-OCRv4_rec.yml -o Global.pretrained_model="./output/yibiao_ocr/rec/best_model/model" Global.save_inference_dir="./output/yibiao_ocr/rec/rec_inference/"
```

识别模型inference 模型预测：

```bash
python3 tools/infer/predict_rec.py --rec_model_dir="./output/yibiao_ocr/rec/rec_inference/" --image_dir="./images/" --rec_image_shape="3, 48, 320" --rec_char_dict_path="ppocr/utils/ppocr_keys_v1.txt" --use_gpu=True
```

Rec模型转onnx：

```Plain Text
paddle2onnx --model_dir ./output/yibiao_ocr/rec/rec_inference --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ./output/yibiao_ocr/rec/rec_inference/yibiao_rec.onnx --opset_version 12 --enable_dev_version True
```
```Plain Text
python -m paddle2onnx.optimize --input_model ./output/yibiao_ocr/rec/rec_inference/yibiao_rec.onnx --output_model ./output/yibiao_ocr/rec/rec_inference/yibiao_rec_op.onnx --input_shape_dict "{'x':[1,3,48,320]}"
```

## 导出RKNN

检测模型转RKNN：

```bash 
cd convert_rknn/det
python convert.py <onnx_model> <TARGET_PLATFORM> <dtype(optional)> <output_rknn_path(optional)>
# 例如: python convert.py ../model/ppocrv4_det.onnx rk3588
# 输出文件保存为: ../model/ppocrv4_det.rknn
```

识别模型转RKNN：

```
cd convert_rknn/rec
python convert.py <onnx_model> <TARGET_PLATFORM> <dtype(optional)> <output_rknn_path(optional)>
# 例如: python convert.py ../model/ppocrv4_rec.onnx rk3588
# 输出文件保存为: ../model/ppocrv4_rec.rknn
```
