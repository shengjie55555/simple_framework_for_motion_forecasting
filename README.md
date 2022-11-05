# Simple_Framework_For_Motion_Forecasting
## 开发流程
1. 数据预处理：处理数据并保存成pickle
   1. 注意数据接口尽量保持一致，会影响后续一系列操作
2. 配置文件：网络、训练和文件保存路径等参数
   1. **新建**```config/cfg_{modelname}.py```文件
3. Dataset接口类：加载预处理保存的文件
   1. 在```utils.dataset.py```中创建：```{ModelName}Dataset```类
4. 网络模型：创建预测模型
   1. **新建**```model/{modelname}.py```文件
   2. 创建```agent_gather & graph_gather```函数
   3. 创建```{ModelName}```类
5. 损失函数：用于计算轨迹误差和置信度误差
   1. 在```model.loss.py```中创建：```{ModelName}Loss```和```{ModelName}PredLoss```类
6. 平均误差和指标
   1. 在```utils.log_utils.py```中创建：```{ModelName}AverageLoss```和```{ModelName}AverageMetrics```类
7. 可视化：绘制目标车辆的预测结果
   1. **新建**```vis_{modelname}.py```文件
   2. 创建```Vis```类

**备注**：
1. 上述通过```args.model```参数和```importlib```来自动化加载所需要的类
2. 优化器目前默认```Adam```
3. 学习率调整策略默认```MultiStepLR```

## 框架中已有模型
| methods   | config                      | dataset                        | model                     | loss                     | average_loss                         | average_metrics                         | vis                         |
|-----------|-----------------------------|--------------------------------|---------------------------|--------------------------|--------------------------------------|-----------------------------------------|-----------------------------|
| ATDS      | config.cfg_atds.config      | utils.dataset.ATDSDataset      | model.atds.ATDS           | model.loss.ATDSLoss      | utils.log_utils.ATDSAverageLoss      | utils.log_utils.ATDSAverageMetrics      | visualize.vis_atds.Vis      |
| VectorNet | config.cfg_vectornet.config | utils.dataset.VectorNetDataset | model.vectornet.VectorNet | model.loss.VectorNetLoss | utils.log_utils.VectorNetAverageLoss | utils.log_utils.VectorNetAverageMetrics | visualize.vis_vectornet.Vis |
| MHL       | config.cfg_mhl.config       | utils.dataset.ATDSDataset      | model.mhl.MHL             | model.loss.MHLLoss       | utils.log_utils.MHLAverageLoss       | utils.log_utils.MHLAverageMetrics       | visualize.vis_mhl.Vis       |

## 复杂场景
考虑场景的不确定性和驾驶员决策的不确定性分成以下几类：

| 类别  | 场景ID                   |
|-----|------------------------|
| 变道  | **9**,239,438          |
| 左转  | 52,356,**553**         |
| 右转  | **146**,402,1001       |
| 路口  | 29,291,**485**,649,710 |
| 加速  | 332,**543**,743        |
| 减速  | 183,**184**,514        |