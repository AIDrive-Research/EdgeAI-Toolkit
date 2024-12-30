**环境安装**

   1. Clone repo and install [requirements.txt](train/classify/resnet18/requirements.txt) in a python>=3.8.0 environment, including pytorch>=1.8

   2. ```bash
      git clone https://github.com/AIDrive-Research/EdgeAI-Toolkit.git
      cd EdgeAI-Toolkit/train/classify/resnet18
      pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
      ```

**数据准备**

数据集结构如下：

```bash
|-dataset
   |-images
      |- 0
         |-xxxxx.jpg
         |-xxxxx.jpg
         |...
      |- 1
         |-xxxxx.jpg
         |-xxxxx.jpg
         |...
      |...
   |-train
      |- 0
         |-xxxxx.jpg
         |-xxxxx.jpg
         |...
      |- 1
         |-xxxxx.jpg
         |-xxxxx.jpg
         |...
      |...
   |-test
      |- 0
         |-xxxxx.jpg
         |-xxxxx.jpg
         |...
      |- 1
         |-xxxxx.jpg
         |-xxxxx.jpg
         |...
      |...
```

**数据划分**

将数据集放置于dataset下的images文件夹下，按照类别进行放置，0文件夹表示第一类，1文件夹表示第二类，依次类推。执行如下代码，进行数据集划分，得到训练集与验证集。注：将`src_path`，`train_path`，`test_path`修改为自己数据集路径。

```bash
python split_train_test.py
```

**模型训练**

将`train.py`配置参数中的`n_classes`修改为自己数据集类别数，将`train_dataset`，`test_dataset`修改为自己的数据路径。执行如下代码。

```
python train.py
```

**模型导出**

将`convert_onnx.py`中的`input_path`，`output_path`分别修改为自己的模型权重路径，`onnx`文件导出路径。执行如下代码。

```bash
python convert_onnx.py
```