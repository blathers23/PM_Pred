### 课设项目：

用 ConvLSTM 实现 $PM_{10}$ 和 $PM_{2.5}$ 的浓度分布预测



#### 公式：

##### ConvLSTM：

$$
I_{t} = Sigmoid(Conv(X_t, W_{xi}) + Conv(H_{t-1}, W_{hi}) + b_i) \\
F_{t} = Sigmoid(Conv(X_t, W_{xf}) + Conv(H_{t-1}, W_{hf}) + b_f) \\
O_{t} = Sigmoid(Conv(X_t, W_{xo}) + Conv(H_{t-1}, W_{ho}) + b_o) \\
G_{t} = Tanh(Conv(X_t, W_{xg}) + Conv(H_{t-1}, W_{hg}) + b_g) \\
C_t = F_t \odot C_{t-1} + I_t \odot G_t \\
H_t = O_t \odot Tanh(C_t) \\ 
Pred = ReLU(Conv(X_t, W_{xx}) + Conv(H_t, W_{xh}) + b_{p})
$$

##### ConvGRU:

$$
R_{t} = Sigmoid(Conv(X_t, W_{xr}) + Conv(H_{t-1}, W_{hr})+b_r) \\
Z_{t} = Sigmoid(Conv(X_t, W_{xz}) + Conv(H_{t-1}, W_{hz})+b_z) \\
\tilde{H}_t = Tanh(Conv(X_t, W_{x \tilde{h}})+Conv(R_t \odot H_{t-1}, W_{r \tilde{h}}) + b_{\tilde{h}}) \\
H_t = Z_t \odot H_{t-1} + (1 - Z_t) \odot \tilde{H} \\
Pred = ReLU(Conv(X_t, W_{xx}) + Conv(H_t, W_{xh}) + b_{p})
$$

#### 进度：

##### update 4/10:

实现了 ConvLSTM 和 ConvGRU 的基本框架，以及训练所需的代码。

