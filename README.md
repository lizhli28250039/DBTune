# DBTune

**1. Architecture and process details**

The architecture of this program is shown in the following figure:

![image](https://github.com/lizhli28250039/DBTune/assets/140188927/08fe1d4a-5c03-4084-b531-d59b5b76a9d8)

As shown in the above figure：The agent obtains the status of the database (including client load information, hardware environment information, and internal database status information) from the database.The agent transfers status data to DBTune.The status data is first determined through a sample filter to determine whether it is suitable for the recommended configuration.If it is not suitable for providing recommended configurations, simply return.If it is suitable for providing recommended configurations, package the data into samples and store them in the sample pool, and output the data to the database configuration Knobs through the Actor network.



The communication between the Agent and DBTune is through the HTTP protocol.DBTune is an HTTP server.The agent is an HTTP client.The data format sent by the agent to DBTune is:
b"{'state': '0.648, 0.514, 0.787, 0.566, 0.591, 0.363, 0.329, 0.024, 0.408, 0.005, 0.224, 0.421, 0.467, 0.91, 0.681, 0.476, 0.083, 0.999, 0.991, 0.216, 0.662, 0.934, 0.095, 0.25, 0.768, 0.309, 0.747, 0.326, 0.519, 0.354, 0.482, 0.2, 0.657, 0.903, 0.481, 0.955, 0.699, 0.053, 0.534, 0.907, 0.993, 0.297, 0.556, 0.835, 0.487, 0.548, 0.782, 0.195, 0.216, 0.306, 0.701, 0.574, 0.785, 0.164, 0.885, 0.347, 0.682, 0.975, 0.661, 0.821, 0.902, 0.789, 0.444, 0.073', 'TPS': '8608'}\r\n"

The Knobs data format returned by DBTune to the Agent is:'[ 0.530  0.406  0.620  0.481  0.421  0.595  0.598  0.415  0.484  0.253  0.595  0.606]'12 data represent 12 database parameters (different database parameters are different)：


![image](https://github.com/lizhli28250039/DBTune/assets/140188927/53b8e1b5-3182-4c80-b249-86d433d2eeb8)

For example, the data returned by DBTune is called the recommended configuration coefficient.The formula for calculating the true configuration value of the database is:
Default configuration+(maximum configuration - minimum configuration) * recommended configuration factor.
Assuming the first parameter returned by DBTune is 0.530, the recommended configuration for database **effective_cache_size** is: 4+(16-4) * 0.530=10.36G

Input of sample filter:
tensor([[0.4870, 0.9650, 0.0650, 0.5410, 0.4660, 0.6010, 0.0890, 0.5790, 0.2700,
         0.5560, 0.6450, 0.4810, 0.3550, 0.2490, 0.9340, 0.4530, 0.5300, 0.0190,
         0.5080, 0.0060, 0.1440, 0.4730, 0.3770, 0.0540, 0.5880, 0.1640, 0.5570,
         0.1440, 0.9370, 0.7710, 0.9570, 0.1410, 0.3050, 0.0400, 0.2770, 0.8070,
         0.1770, 0.1550, 0.9550, 0.1550, 0.8340, 0.0410, 0.3860, 0.3500, 0.3420,
         0.8160, 0.4760, 0.7830, 0.4710, 0.8170, 0.8820, 0.4400, 0.7810, 0.8150,
         0.2960, 0.1240, 0.1860, 0.4360, 0.1190, 0.5300, 0.8290, 0.4850, 0.8180,
         0.6560]])
Output of sample filter:
tensor[[0.0318]]


[Note]: For the convenience of program debugging, the Agent in this code does not use real environment data to obtain the status data and TPS data of the database.We will place the status data and TPS data in metric.txt and TPS.txt.Actual readers should build a real database environment.Both the status data and TPS data are obtained under actual workloads, and the actual database status data and TPS data are obtained.For specific data formats, please refer to the data formats in metric. txt and TPS. txt in this code.
