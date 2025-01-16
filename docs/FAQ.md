### 1、如何确定设备是否支持自定义算法？

- 版本 ≥3.2.1，支持自定义推理代码&后处理代码
- 版本＜3.2.1,支持自定义后处理代码

![faq_1](./assets/faq_1.png)



### 2、如何查看产品型号？

下图系统设置-设备升级中的ks968为产品型号

![faq_1](./assets/faq_1.png)

### 3、模型量化中platform如何确定？

- 版本中ks968，platform为rk3588
- 版本中ks916，platform为rk3568

### 4、设备中的全部示例算法包在哪下载？

[点此下载]( https://pan.baidu.com/s/1xNDw1ns9Ra90xlOfP9LF1A?pwd=0000 )。下载对应产品型号，对应语言、对应版本的demo文件夹。如我购买的产品是中文版ks968-m2-3.2.1，则下载：ks968-m1&m2->标准算法仓->中文算法包->V3.2.1->demo文件夹。

![faq_5](./assets/faq_5.png)

### 5、如何调试代码&查看日志？

- 在下图所示红色框内，连续点击7次，打开开发者模式（版本≥3.1.6具备此功能）

![faq_2](./assets/faq_2.png)

- 在高级设置，终端管理中，可进入盒子后台调试&查看日志（请勿删除系统源码，谨慎操作，否则造成设备不可用）

![faq_3](./assets/faq_3.png)

- 调试代码。导入`logger`包，使用`LOGGER.info`输出日志。示例如下。

```python
from logger import LOGGER

LOGGER.info('boxes:{},classes:{},scores:{}'.format(boxes, classes, scores))
```

- 查看日志。

查看推理模块日志。

```bash
tail -f ks/ks968/data/logs/engine/0/engine.log
```

查看后处理模块日志。

```bash
tail -f ks/ks968/data/logs/filter/filter.log
```

![faq_3](./assets/faq_4.png)
