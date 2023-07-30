# DBTune
本程序的架构如下图所示：

![image](https://github.com/lizhli28250039/DBTune/assets/140188927/08fe1d4a-5c03-4084-b531-d59b5b76a9d8)


Agent从数据库获取数据库的状态（包括客户端负载信息、硬件环境信息、数据库内部状态信息），将状态数据传送给test_ddpg，首先通过样本过滤器判断该数据是否适合推荐配置，如果不适合，直接返回，如果适合则将数据打包成样本存放入样本池，并且将数据通过Actor网络输出数据库配置Knobs。

Agent与DBTune之间通信通过HTTP协议，DBTune为HTTP服务端，Agent为HTTP客户端，Agent发送给DBTune的数据格式为：
b"{'state': '0.648, 0.514, 0.787, 0.566, 0.591, 0.363, 0.329, 0.024, 0.408, 0.005, 0.224, 0.421, 0.467, 0.91, 0.681, 0.476, 0.083, 0.999, 0.991, 0.216, 0.662, 0.934, 0.095, 0.25, 0.768, 0.309, 0.747, 0.326, 0.519, 0.354, 0.482, 0.2, 0.657, 0.903, 0.481, 0.955, 0.699, 0.053, 0.534, 0.907, 0.993, 0.297, 0.556, 0.835, 0.487, 0.548, 0.782, 0.195, 0.216, 0.306, 0.701, 0.574, 0.785, 0.164, 0.885, 0.347, 0.682, 0.975, 0.661, 0.821, 0.902, 0.789, 0.444, 0.073', 'TPS': '8608'}\r\n"

DBTune返回给Agent的Knobs数据格式为：'[ 0.530  0.406  0.620  0.481  0.421  0.595  0.598  0.415  0.484  0.253  0.595  0.606]'12个数据分别代表12个数据库参数（不同数据库参数不一样）：


![image](https://github.com/lizhli28250039/DBTune/assets/140188927/254b6f3a-1fb4-43ae-98e4-053278ea4994)


样本过滤器的输入：
tensor([[0.4870, 0.9650, 0.0650, 0.5410, 0.4660, 0.6010, 0.0890, 0.5790, 0.2700,
         0.5560, 0.6450, 0.4810, 0.3550, 0.2490, 0.9340, 0.4530, 0.5300, 0.0190,
         0.5080, 0.0060, 0.1440, 0.4730, 0.3770, 0.0540, 0.5880, 0.1640, 0.5570,
         0.1440, 0.9370, 0.7710, 0.9570, 0.1410, 0.3050, 0.0400, 0.2770, 0.8070,
         0.1770, 0.1550, 0.9550, 0.1550, 0.8340, 0.0410, 0.3860, 0.3500, 0.3420,
         0.8160, 0.4760, 0.7830, 0.4710, 0.8170, 0.8820, 0.4400, 0.7810, 0.8150,
         0.2960, 0.1240, 0.1860, 0.4360, 0.1190, 0.5300, 0.8290, 0.4850, 0.8180,
         0.6560]])
样本过滤器的输出：
tensor[[0.0318]]
