**环境安装**

1. 系统要求

- 操作系统：Ubuntu

2. [下载](https://pan.baidu.com/s/1AVKxKGzBEc0iBMuSMJaDJQ?pwd=0000)量化套件，提供了python3.8的量化环境whl文件，根据设备的软件版本进行安装

- v3.1.1之前的版本（不包含v3.1.1）

  ```bash
    cd rknn-toolkit2
    conda create -n py38-rk1.6 python=3.8
    conda activate py38-rk1.6
    pip3 install -i https://pypi.doubanio.com/simple/ tf-estimator-nightly==2.8.0.dev2021122109
    pip3 install rknn_toolkit2-1.6.0+81f21f4d-cp38-cp38-linux_x86_64.whl
  ```

- v3.1.1之后的版本（包含v3.1.1）

  ```bash
    cd rknn-toolkit2
    conda create -n py38-rk2.2 python=3.8
    conda activate py38-rk2.2
    pip3 install rknn_toolkit2-2.2.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  ```

**模型量化**

1. 在训练集中随机选取图片进行模型量化，精度校准，数量在80-120之间，目录结构如下：

   ```
    images:
    	xxx.jpg
   ```

2. 把图片路径保存至xxx.txt

   ```
    find ./images/ -name "*.jpg">custom.txt
   ```

3. 模型量化

   修改convert.py：

   - DATASET_PATH：量化图片路径

   运行：

   ```
    python convert.py onnx_model_path platform i8/fp output_rknn_path
   ```

   其中：

   - onnx_model_path：训练后导出的onnx模型文件位置
   - platform：[rk3562,rk3566,rk3568,rk3588]
   - i8/fp：i8代表使用图片量化；fp代表不量化
   - output_rknn_path：量化后模型的保存路径
