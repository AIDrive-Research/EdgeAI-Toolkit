### 算法包构成

算法包是导入智能分析设备的最终文件。包括模型文件与后处理文件两部分。各部分详情如下。

<img src="..\docs\assets\structure.png" alt="structure.png" style="zoom:55%;" />

- **① person_intrusion**: 算法名称，可修改为自定义的算法名称。
- **② model:** 用于存放模型文件、推理代码以及模型配置文件，**名称不可修改**。
- **③ postprocessor:** 用于存放算法后处理代码、算法配置文件与前端配置文件，**名称不可修改**。
- **④ person:** 模型名称，用于存放量化后的模型文件，名称可修改为自定义的算法名称。
- **⑤ detect.py:** 算法推理代码，名称可修改为自定义的算法名称。
- **⑥ model.yaml:** 模型配置文件，**名称不可修改**。
- **⑦ person_intrusion.json:** 前端配置文件，名称可修改为自定义的算法名称。
- **⑧ person_intrusion.py:** 算法后处理代码，名称可修改为自定义的算法名称。
- **⑨ postprocessor.yaml:** 算法后处理配置文件，**名称不可修改。**

<img src="..\docs\assets\algname.png" alt="algname.png" style="zoom:75%;" />

- **算法名称一致：** 算法包文件夹名称、前端配置文件名称、算法后处理代码文件名称、算法后处理文件中算法名称需保持一致。

<img src="..\docs\assets\modelname.png" alt="model.png" style="zoom:65%;" />

- **模型名称一致：** 模型文件夹名称、模型配置文件、算法后处理代码文件、前端配置文件中的模型名称需保持一致。