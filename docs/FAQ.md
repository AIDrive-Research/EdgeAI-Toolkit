### 1、如何确定设备是否支持自定义算法？

- 版本≥3.2.1，支持自定义推理代码&后处理代码
- 版本＜3.2.1,支持自定义后处理代码

![faq_1](./assets/faq_1.png)

### 2、模型量化中platform如何确定？

- 版本中ks968，platform为rk3588
- 版本中ks916，platform为rk3568

### 3、如何调试代码&查看日志？

- 在下图所示红色框内，连续点击7次，打开开发者模式

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
