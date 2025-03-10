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

## 模型选择
建议选择PP-OCRv4模型（配置文件：[ch_PP-OCRv4_det_student.yml](https://github.com/AIDrive-Research/EdgeAI-Toolkit/tree/main/train/ocr/paddleocr/configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_cml.yml)，预训练模型：[ch_PP-OCRv4_det_train.tar](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_train.tar)）进行微调，其精度与泛化性能是目前提供的最优预训练模型。
注意：在使用上述预训练模型的时候，需要使用文件夹中的student.pdparams文件作为预训练模型，即，仅使用学生模型。

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

1. 单卡训练

   ```python
   python3 tools/train.py -c configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_student.yml -o Global.pretrained_model=./pretrain_models/ch_PP-OCRv4_det_train/best_accuracy.pdparams
   ```

2. 推荐使用多卡训练

   ```python
   python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_student.yml -o Global.pretrained_model=./pretrain_models/ch_PP-OCRv4_det_train/best_accuracy.pdparams
   ```

## 模型导出
1. ONNX导出，支持RK3566, RK3568, RK3588, RK3562, RK1808, RV1109, RV1126

- 安装paddle2onnx
   ```bash 
   pip install paddle2onnx
   ```
- Det转换onnx
   ```bash
   paddle2onnx 
   --model_dir ./model/ch_PP-OCRv4_det_infer 
   --model_filename inference.pdmodel 
   --params_filename inference.pdiparams 
   --save_file ./model/ch_PP-OCRv4_det_infer/model.onnx 
   --opset_version 12 
   --enable_dev_version True
   ```
   ```bash
   python -m paddle2onnx.optimize 
   --input_model model/ch_PP-OCRv4_det_infer/model.onnx                                  
   --output_model model/ch_PP-OCRv4_det_infer/ppocrv4_det.onnx 
   --input_shape_dict "{'x':[1,3,480,480]}"
   ```
- Rec转换onnx
   ```bash
   paddle2onnx 
   --model_dir ./model/ch_PP-OCRv4_rec_infer 
   --model_filename inference.pdmodel 
   --params_filename inference.pdiparams 
   --save_file ./model/ch_PP-OCRv4_rec_infer/model.onnx 
   --opset_version 12 
   --enable_dev_version True
   ```
   ```bash
   python -m paddle2onnx.optimize 
   --input_model model/ch_PP-OCRv4_rec_infer/model.onnx 
   --output_model model/ch_PP-OCRv4_rec_infer/ppocrv4_rec.onnx --input_shape_dict "{'x':[1,3,48,320]}"
   ```

2. RKNN转换
   ```bash 
   cd ocr/convert_rknn/det
   python convert.py <onnx_model> <TARGET_PLATFORM> <dtype(optional)> <output_rknn_path(optional)>
   # 例如: python convert.py ../model/ppocrv4_det.onnx rk3588
   # 输出文件保存为: ../model/ppocrv4_det.rknn
   ```

