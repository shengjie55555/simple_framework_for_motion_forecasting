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
2. 各个模型之间的差异：数据接口、配置文件、模型结构
3. 优化器目前默认```Adam```
4. 学习率调整策略默认```MultiStepLR```

## 框架中已有模型
| methods   | config                      | dataset                        | model                     | loss                     | average_loss                         | average_metrics                         | vis                         |
|-----------|-----------------------------|--------------------------------|---------------------------|--------------------------|--------------------------------------|-----------------------------------------|-----------------------------|
| ATDS      | config.cfg_atds.config      | utils.dataset.ATDSDataset      | model.atds.ATDS           | model.loss.ATDSLoss      | utils.log_utils.ATDSAverageLoss      | utils.log_utils.ATDSAverageMetrics      | visualize.vis_atds.Vis      |
| VectorNet | config.cfg_vectornet.config | utils.dataset.VectorNetDataset | model.vectornet.VectorNet | model.loss.VectorNetLoss | utils.log_utils.VectorNetAverageLoss | utils.log_utils.VectorNetAverageMetrics | visualize.vis_vectornet.Vis |
| MHL       | config.cfg_mhl.config       | utils.dataset.ATDSDataset      | model.mhl.MHL             | model.loss.MHLLoss       | utils.log_utils.MHLAverageLoss       | utils.log_utils.MHLAverageMetrics       | visualize.vis_mhl.Vis       |
| LaneGCN   | config.cfg_lanegcn.config   | utils.dataset.LaneGCNDataset   | model.lanegcn.LaneGCN     | model.loss.LaneGCNLoss   | utils.log_utils.LaneGCNAverageLoss   | utils.log_utils.LaneGCNAverageMetrics   | visualize.vis_lanegcn.Vis   |
| DS        | config.cfg_ds.config        | utils.dataset.DSDataset        | model.ds.DS               | model.loss.DSLoss        | utils.log_utils.DSAverageLoss        | utils.log_utils.DSAverageMetrics        | visualize.vis_ds.Vis        |

## 复杂场景
考虑场景的不确定性和驾驶员决策的不确定性分成以下几类：
* ATDSDataset中验证集的argo_id和Argoverse中val的file_name不对应
* 可以通过orig的坐标来对齐

| 类别  | 场景ID                                                          |
|-----|---------------------------------------------------------------|
| 变道  | **9**,239,438                                                 |
| 左转  | 52,356,**553**                                                |
| 右转  | **146**,402,1001                                              |
| 路口  | 29,291,**485**,649,710,2112,3933,4563,10057,13467,13604,19567 |
| 加速  | 332,**543**,743                                               |
| 减速  | 183,**184**,514                                               |

## 验证集结果
| methods       | minADE_1 | minFDE_1 | MR_1   | minADE_6 | minFDE_6 | MR_6   | brier-minFDE |
|---------------|----------|----------|--------|----------|----------|--------|--------------|
| MHL           | 1.5562   | 3.4983   | 0.5550 | 0.8032   | 1.3291   | 0.1568 | 1.9275       |
| MHL+VectorNet | 1.4705   | 3.2696   | 0.5355 | 0.7478   | 1.1725   | 0.1250 | 1.7821       |
| MHL+LaneGCN   | 1.4344   | 3.1791   | 0.5173 | 0.7312   | 1.1146   | 0.1101 | 1.7227       |
| MHL+DS        | 1.4448   | 3.2107   | 0.5333 | 0.7276   | 1.1228   | 0.1089 | 1.7252       |
| ATDS-v2.0     | 1.3486   | 2.9541   | 0.4916 | 0.7059   | 1.0604   | 0.0991 | 1.6550       |
| ATDS-v4.2     | 1.3214   | 2.8773   | 0.4836 | 0.7015   | 1.0458   | 0.0988 | 1.6333       |
| ATDS-v4.3     | 1.3084   | 2.8709   | 0.4817 | 0.6936   | 1.0315   | 0.0953 | 1.6196       |