#
# DBTune - test_ddpg.py
#
# Copyright (c) 2022-7, Carnegie Mellon University Database Group
#


from ddpg import DDPG

import json
import requests

import torch#深度学习的pytoch平台
import torch.nn as nn
import numpy as np
import random
import linecache
import time#可以用来简单地记录时间


#随机种子
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

n_states = 64  #The sum of the quantities of , , and  dimensions
sample_filter_data_path = "sample_filter_data_path"




class SampleFilter(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [64, 128, 64, 32, 1]  # 网络每一层的神经元个数，[1,10,1]说明只有一个隐含层，输入的变量是一个，也对应一个输出。如果是两个变量对应一个输出，那就是[2，10，1]
        self.layer1 = nn.Linear(layers[0],
                                layers[1])  # 用torh.nn.Linear构建线性层，本质上相当于构建了一个维度为[layers[0],layers[1]]的矩阵，这里面所有的元素都是权重
        self.layer2 = nn.Linear(layers[1], layers[2])
        self.layer3 = nn.Linear(layers[2], layers[3])
        self.layer4 = nn.Linear(layers[3], layers[4])
        self.elu = nn.ELU()  # 非线性的激活函数。如果只有线性层，那么相当于输出只是输入做了了线性变换的结果，对于线性回归没有问题。但是非线性回归我们需要加入激活函数使输出的结果具有非线性的特征

        x = np.linspace(-np.pi, np.pi).astype(np.float32)
        y = np.sin(x)
        # 随机取25个点
        self.x_train = random.sample(x.tolist(), 25)  # x_train 就相当于网络的输入
        self.y_train = np.sin(self.x_train)  # y_train 就相当于输入对应的标签，每一个输入都会对应一个标签
        #plt.scatter(self.x_train, self.y_train, c="r")
        #plt.plot(x, y)
        # plt.show()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.gline = 1
        self.file = 0

    def getdata(self, line):
        if line != 0:
            self.gline = line

        # print("gline ", self.gline)

        if self.gline == 1:
            with open(sample_filter_data_path, 'r') as file:
                self.file = file

        line_content = linecache.getline(sample_filter_data_path, self.gline)
        self.gline = self.gline + 1

        rep = [('[', ''), (']', ''), ('\n', ''), ('\'', '\"')]

        for c, r in rep:
            if c in line_content:
                line_content = line_content.replace(c, r)

        label = line_content.split(",")[-1]

        str_index = line_content.rfind(',')  # 获取最后一个点的下标

        state_results = line_content[:str_index]
        state_array = self.stringarr2floatarray(state_results)

        x = np.linspace(-np.pi, np.pi).astype(np.float32)
        y = np.sin(x)
        x_train = random.sample(x.tolist(), 25)
        self.y_train = np.sin(self.x_train)  # y_train 就相当于输入对应的标签，每一个输入都会对应一个标签

        return state_array, int(label)

    def forward(self, d):  # d就是整个网络的输入
        d1 = self.layer1(d)
        d1 = self.elu(d1)  # 每一个线性层之后都需要加入一个激活函数使其非线性化。
        d2 = self.layer2(d1)  # 但是在网络的最后一层可以不用激活函数，因为有些激活函数会使得输出结果限定在一定的值域里。
        d2 = self.elu(d2)  # 每一个线性层之后都需要加入一个激活函数使其非线性化。
        d3 = self.layer3(d2)
        d3 = self.elu(d3)  # 每一个线性层之后都需要加入一个激活函数使其非线性化。
        d4 = self.layer4(d3)
        return d4

    def trainnn(self):
        device = self.device  # 在跑深度学习的时候最好使用GPU，这样速度会很快。不要的话默认用cpu跑
        epochs = 10000  # 这是迭代次数，把所有的训练数据输入到网络里去就叫完成了一次epoch。
        learningrate = 1e-4  # 学习率，相当于优化算法里的步长，学习率越大，网络参数更新地更加激进。学习率越小，网络学习地更加稳定。
        net = SampleFilter().to(device=device)  # 网络的初始化
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=learningrate)  # 优化器，不同的优化器选择的优化方式不同，这里用的是随机梯度下降SGD的一种类型，Adam自适应优化器。需要输入网络的参数以及学习率，当然还可以设置其他的参数
        mseloss = nn.MSELoss()  # 损失函数，这里选用的是MSE。损失函数也就是用来计算网络输出的结果与对应的标签之间的差距，差距越大，说明网络训练不够好，还需要继续迭代。
        MinTrainLoss = 1e10
        train_loss = []  # 用一个空列表来存储训练时的损失，便于画图

        start = time.time()
        start0 = time.time()
        for epoch in range(1, epochs + 1):

            x_state, y_label = self.getdata(line=0)

            # print("x_state:", x_state)
            # print("y_label:", y_label)

            pt_x_train = torch.from_numpy(np.array(x_state)).to(device=device, dtype=torch.float32).reshape(1,
                                                                                                            -1)  # 这里需要把我们的训练数据转换为pytorch tensor的类型，并且把它变成gpu能运算的形式。
            pt_y_train = torch.from_numpy(np.array(y_label)).to(device=device, dtype=torch.float32).reshape(1,
                                                                                                            -1)  # reshap的目的是把维度变成(25,1),这样25相当于是batch，我们就可以一次性把所有的点都输入到网络里去，最后网络输出的结果也不是(1,1)而是(25,1)，我们就能直接计算所有点的损失

            net.train()  # net.train()：在这个模式下，网络的参数会得到更新。对应的还有net.eval()，这就是在验证集上的时候，我们只评价模型，并不对网络参数进行更新。
            pt_y_pred = net(pt_x_train)  # 将tensor放入网络中得到预测值
            loss = mseloss(pt_y_pred, pt_y_train)  # 用mseloss计算预测值和对应标签的差别
            optimizer.zero_grad()  # 在每一次迭代梯度反传更新网络参数时，需要把之前的梯度清0，不然上一次的梯度会累积到这一次。
            loss.backward()  # 反向传播
            optimizer.step()  # 优化器进行下一次迭代
            if epoch % 10 == 0:  # 每10个epoch保存一次loss
                end = time.time()
                print("epoch:[%5d/%5d] time:%.2fs current_loss:%.5f"
                      % (epoch, epochs, (end - start), loss.item()))
                start = time.time()
            train_loss.append(loss.item())
            if train_loss[-1] < MinTrainLoss:
                torch.save(net.state_dict(), "model.pth")  # 保存每一次loss下降的模型
                MinTrainLoss = train_loss[-1]
        end0 = time.time()
        print("训练总用时: %.2fmin" % ((end0 - start0) / 60))

    def makedata(self):
        open(sample_filter_data_path, 'w')
        with open(sample_filter_data_path, 'a')as file:
            for i in range(100000):
                state = [round(random.uniform(0, 1), 3) for _ in range(64)]
                random_list = [0, 1]
                random_element = random.choice(random_list)
                content = str(state) + ',' + str(random_element)
                file.write(content)
                content = '\n'
                file.write(content)

    def stringarr2floatarray(self, string_arr):
        state_list = string_arr.split(',')
        state_arr = np.array(state_list)

        ini_array = ', '.join(state_arr)
        ini_array = np.fromstring(ini_array, dtype=np.float64, sep=', ')
        new_array = ini_array.astype('float')

        return new_array

    def filter(self, x_state):
        pt_x_test = torch.from_numpy(np.array(x_state)).to(device=self.device, dtype=torch.float32).reshape(1, -1)
        DBnn = SampleFilter().to(self.device)
        DBnn.load_state_dict(torch.load("model.pth", map_location=self.device))  # pytorch 导入模型
        DBnn.eval()  # 这里指评价模型，不反传，所以用eval模式
        predict_y_test = DBnn(pt_x_test)

        if predict_y_test > 0.5:
            return 1
        else:
            return 0

    def test(self):

        x_state, y_label = self.getdata(line=22)

        print("x_state:", x_state)
        print("y_label:", y_label)

        # print("self.x_train:", self.x_train)
        # print("self.y_train:", self.y_train)

        pt_x_test = torch.from_numpy(np.array(x_state)).to(device=self.device, dtype=torch.float32).reshape(1,
                                                                                                            -1)  # 这里需要把我们的训练数据转换为pytorch tensor的类型，并且把它变成gpu能运算的形式。
        pt_y_test = torch.from_numpy(np.array(y_label)).to(device=self.device, dtype=torch.float32).reshape(1,
                                                                                                            -1)  # reshap的目的是把维度变成(25,1),这样25相当于是batch，我们就可以一次性把所有的点都输入到网络里去，最后网络输出的结果也不是(1,1)而是(25,1)，我们就能直接计算所有点的损失
        print("pt_x_test:", pt_x_test)

        DBnn = SampleFilter().to(self.device)
        DBnn.load_state_dict(torch.load("model.pth", map_location=self.device))  # pytorch 导入模型
        DBnn.eval()  # 这里指评价模型，不反传，所以用eval模式
        predict_y_test = DBnn(pt_x_test)

        print("pt_y_test:", pt_y_test)
        print("predict_y_test:", predict_y_test)

        y_test = pt_y_test.detach().cpu().numpy()  # 输出结果torch tensor，需要转化为numpy类型来进行可视化




class DBTune():

    def __init__(self):
        print("DBTune init")
        self.ddpg = DDPG(n_actions=12, n_states=64, gamma=0, alr=0.02)
        self.state_old = []
        self.sample_num = 1
        self.tps_old = 10000


    def stringarr2floatarray(self,string_arr):
        state_list = string_arr.split(',')
        state_arr = np.array(state_list)

        ini_array = ', '.join(state_arr)
        ini_array = np.fromstring(ini_array, dtype=np.float64, sep=', ')
        new_array = ini_array.astype('float')

        return new_array


    def Add_Small_Sample_Pool(self,state,reward):
        #print("Add_Small_Sample_Pool begin")
        knob_data = self.ddpg.choose_action(state)

        #print("The Sample number is ",self.sample_num)
        self.sample_num = self.sample_num + 1

        print("knob_data is", knob_data)
        print("reward is", reward)
        #print("state_new is", state)
        #print("state_old is", self.state_old)

        self.SendHTTP(type="POST", data=str(knob_data))

        print("begin add sample.number= ",self.sample_num)

        if len(self.state_old) == 0:
            self.state_old = state
            return
        self.ddpg.add_sample(self.state_old, knob_data, reward, state)
        self.state_old = state


    def get_reward(self,TPS):
        print("get_stata_reward begin",TPS)

        reward = TPS
        return reward

    def SendHTTP(self,type,data):
        url = 'http://127.0.0.1:8080/'

        headers = {
            'Authorization': 'cfe7mpr2fperuifn65g0',
            'Content-Type': 'application/json',
        }
        if type == "GET":
            payload = {}
            sendBody = json.dumps(payload)
            try:
                response = requests.get(url, headers=headers, data=sendBody)
            except requests.exceptions.ConnectionError:
                print("request timeout.url=", url)
                return False
        else:
            payload = {}
            payload["data"] = data
            sendBody = json.dumps(payload)
            try:
                response = requests.post(url, headers=headers, data=sendBody)
            except requests.exceptions.ConnectionError:
                print("request timeout.url=", url)
                return False

        return response





    def get_response(self):
        response = self.SendHTTP(type = "GET",data = "")

        text = ''
        for i in response.content:
            text = text + chr(i)

        for i in text:
            if i == '\r':
                text = text.replace(i, '')
            if i == '\n':
                text = text.replace(i, '')

        text = text.replace('\'', '"')
        json_data = json.loads(text)
        state = json_data['state']
        state_new = self.stringarr2floatarray(state)
        TPS = json_data["TPS"]
        if TPS == '':
            print("TPS is null!!!!!!")
            exit(0)

        reward = self.get_reward(TPS)
        reward_f = float(reward)
        sample_filter = SampleFilter()
        result = sample_filter.filter(state_new)
        print("filter result:", result)
        if result == 0:
            self.Add_Small_Sample_Pool(state_new,reward_f)


    def aaa(self):
        jsonString = '{"state": "0.14739051889312016, 0.177431486232265", "TPS": 12149}'

        json_data = json.loads(jsonString)
        print("sssssss", json_data)

        state = json_data['state']
        TPS = json_data["TPS"]
        print("state", state)
        print("TPS", TPS)

    def bbb_ddpg_ypreds(self):
        total_reward = 0.0
        print("test_ddpg_ypreds ------------------")
        for _ in range(2):
            prev_metric_data = np.array([random.random()])
            knob_data = self.ddpg.choose_action(prev_metric_data)
            reward = 1.0 if (prev_metric_data[0] - 0.5) * (knob_data[0] - 0.5) > 0 else 0.0
            total_reward += reward
            print("knob_data",knob_data)
            print("reward", reward)





#samplefileter = SampleFilter()
#samplefileter.makedata()
#samplefileter.getdata()
#samplefileter.trainnn()
#samplefileter.test()



client = DBTune()
for i in range(2):
    client.get_response()





