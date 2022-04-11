### 课设项目：

用 ConvLSTM 实现 $PM_{10}$ 和 $PM_{2.5}$ 的浓度分布预测



#### 公式：

##### ConvLSTM：

> 与 ConvLSTM 论文中的实现方式不同，以《动手学深度学习》中的描述为准

$$
I_{t} = Sigmoid(Conv(X_t, W_{xi}) + Conv(H_{t-1}, W_{hi}) + b_i) \\
F_{t} = Sigmoid(Conv(X_t, W_{xf}) + Conv(H_{t-1}, W_{hf}) + b_f) \\
O_{t} = Sigmoid(Conv(X_t, W_{xo}) + Conv(H_{t-1}, W_{ho}) + b_o) \\
\tilde{C}_{t} = Tanh(Conv(X_t, W_{xc}) + Conv(H_{t-1}, W_{hc}) + b_h) \\
C_t = F_t \odot C_{t-1} + I_t \odot \tilde{C}_t \\
H_t = O_t \odot Tanh(C_t) \\ 
Pred = ReLU(Conv(X_t, W_{xx}) + Conv(H_t, W_{xh}) + b_{p}) \\
$$

##### ConvGRU:

$$
R_{t} = Sigmoid(Conv(X_t, W_{xr}) + Conv(H_{t-1}, W_{hr})+b_r) \\
Z_{t} = Sigmoid(Conv(X_t, W_{xz}) + Conv(H_{t-1}, W_{hz})+b_z) \\
\tilde{H}_t = Tanh(Conv(X_t, W_{xh})+Conv(R_t \odot H_{t-1}, W_{rh}) + b_{h}) \\
H_t = Z_t \odot H_{t-1} + (1 - Z_t) \odot \tilde{H} \\
Pred = ReLU(Conv(X_t, W_{xx}) + Conv(H_t, W_{xh}) + b_{p}) \\
$$

#### 进度：

##### update 4/10:

实现了 ConvLSTM 和 ConvGRU 的基本框架，以及训练所需的代码

##### update 4/11:

1. 实现了 `use_random_iter = False` 的训练方式
2. 实现了运行自动保存日志的功能
3. 实现了保存和加载功能
4. 优化了随机采样功能的逻辑
5. 完善了注释



#### 任务：

##### 关键任务：

- [ ] 学习 WRF
- [ ] 生成原始数据
- [ ] 训练模型
- [ ] 撰写论文

##### 闲的没事：

- [ ] 实现 Transformer 版本的网络



#### 参考：

\[ 1 \]: [NIPS 2015: Convolutional LSTM Network](https://proceedings.neurips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf )

\[ 2 \]: [动手学深度学习](https://zh.d2l.ai/)



