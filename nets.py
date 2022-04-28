import torch 
import torch.nn as nn 

# ----------------------------------------------------------- #
# ConvGRU

class ConvGRUCell(nn.Module):
    r"""
    初始化 ConvGRU cell

    Parameters:
    -----------
    input_dim: int
        输入tensor的channel
    hidden_dim: int
        隐状态的channel
    kernel_size: int
        卷积核的size
    bias: bool
        是否添加bias
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super().__init__()
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        
        self.kernel_size = kernel_size
        self.padding = 'same'
        self.bias = bias

        self.batchnorm1 = nn.BatchNorm2d(2*self.hidden_dim)
        self.batchnorm2 = nn.BatchNorm2d(self.hidden_dim)

        self.conv1 = nn.Conv2d(in_channels=self.input_dim+self.hidden_dim,
                                out_channels=2*self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)
        self.conv2 = nn.Conv2d(in_channels=input_dim+self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)

    def forward(self, input_tensor, cur_state):

        h_cur = cur_state

        combined1 = torch.cat([input_tensor, h_cur], dim=1)  # concat alone channel axis

        # combined_conv = self.conv1(combined1)
        combined_conv = self.batchnorm1(self.conv1(combined1))
        
        cc_r, cc_z = torch.split(combined_conv, self.hidden_dim, dim=1)
        r = torch.sigmoid(cc_r)
        z = torch.sigmoid(cc_z)

        combined2 = torch.cat([input_tensor, r * h_cur], dim=1)
        
        # h_next_ = self.conv2(combined2)
        h_next_ = self.batchnorm2(self.conv2(combined2))

        h_next = z * h_cur + (torch.ones_like(z) - z) * h_next_

        return h_next      

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv1.weight.device)

class ConvGRU(nn.Module):
    r"""
    Parameters:
    -------------
    input_dim: int
        输入channel
    hidden_dim: int or list
        隐状态channel
    kernel_size: int or list
        卷积核size
    num_layers: int
        LSTM层的数量
    batch_first: bool
        输入数据的第0维是否是batch
    bias: bool 
        卷积中是否加入偏置

    Input:
    -------------
        (b, t, c, h, w) or (t, b, c, h, w)

    Output:
    -------------
        预测结果: (b, h, w)
        
    Example:
    -------------
        >> x = torch.rand((32, 10, 64, 128, 128)) 

        >> convgru = ConvGRU(64, 16, 3, 3, 3, True, True) 

        >> pred, state = convgru(x) 
    """ 
    def __init__(self, input_dim, hidden_dim, kernel_size, dec_kernel_size, num_layers,
                batch_first=False, bias=True):
        super().__init__()

        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim 
        self.kernel_size = kernel_size 
        self.dec_kernel_size = dec_kernel_size
        self.num_layers = num_layers 
        self.batch_first = batch_first
        self.bias = bias 

        self.padding = 'same'

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvGRUCell(input_dim=cur_input_dim, 
                                hidden_dim=self.hidden_dim[i],
                                kernel_size=self.kernel_size[i],
                                bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

        self.dec = nn.Conv2d(in_channels=self.hidden_dim[-1]+1,
                            out_channels=1, kernel_size=self.dec_kernel_size, 
                            padding=self.padding, bias=self.bias)

        self.relu = nn.ReLU()
    
    def forward(self, input_tensor, hidden_state=None):
        r"""
        Parameters:
        -----------
        input_tensor: 
            5-D tensor (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state:
            None or list of h
        
        Returns:
        ----------
        Pred:
            3-D tensor (b, h, w)
        hidden_state:
            None or list of h
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvGRU
        if hidden_state is not None:
            use_random_iter = False
        else:
            use_random_iter = True
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        state_list = []

        for layer_idx in range(self.num_layers):

            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                    cur_state=h)
                output_inner.append(h)
            
            if not use_random_iter:
                state_list.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

        pred = self.relu(self.dec(torch.cat([input_tensor[:, -1, 0, :, :].unsqueeze(dim=1), layer_output[:, -1, :, :, :]], dim=1)))

        if len(state_list) == 0:
            return pred[:, 0, :, :], None
        else:
            return pred[:, 0, :, :], state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod   # 静态方法无需实例化
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or isinstance(kernel_size, int) or (isinstance(kernel_size, list)
            and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('kernel_size must be int or tuple or list of tuples.')
    
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    def summary(self, PATH):
        with open(PATH + "\\log.txt", "a") as f:
            f.write("\n--------MODEL_INFO--------\n")
            f.write("model:\tConvGRU\n")
            f.write("hidden_dim:\t" + str(self.hidden_dim) + "\n")
            f.write("kernel_size:\t" + str(self.kernel_size) + "\n")
            f.write("dec_kernel_size:\t" + str(self.dec_kernel_size) + "\n")
            f.write("num_layers:\t" + str(self.num_layers) + "\n")

# ----------------------------------------------------------- #
# ConvLSTM

class ConvLSTMCell(nn.Module):
    r"""
    初始化 ConvLstm cell

    Parameters:
    -----------
    input_dim: int
        输入tensor的channel
    hidden_dim: int
        隐状态的channel
    kernel_size: int
        卷积核的size
    bias: bool
        是否添加bias
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super().__init__()
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        
        self.kernel_size = kernel_size
        self.padding = 'same'
        self.bias = bias

        self.batchnorm = nn.BatchNorm2d(4*self.hidden_dim)

        self.conv = nn.Conv2d(in_channels=self.input_dim+self.hidden_dim,
                                out_channels=4*self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)
        
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concat alone channel axis
        
        # combined_conv = self.conv(combined)
        combined_conv = self.batchnorm(self.conv(combined))
        
        cc_i, cc_f, cc_o, cc_c = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        c_next_ = torch.tanh(cc_c)

        c_next = f * c_cur + i * c_next_
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    r"""
    Parameters:
    -------------
    input_dim: int
        输入channel
    hidden_dim: int or list
        隐状态channel
    kernel_size: int or list
        卷积核size
    dec_kernel_size: int
        解码器卷积核size
    num_layers: int
        LSTM层的数量
    batch_first: bool
        输入数据的第0维是否是batch
    bias: bool 
        卷积中是否加入偏置

    Input:
    -------------
        (b, t, c, h, w) or (t, b, c, h, w)

    Output:
    -------------
        预测结果: (b, h, w)
        
    Example:
    -------------
        >> x = torch.rand((32, 10, 64, 128, 128)) 

        >> convlstm = ConvLSTM(64, 16, 3, 3, 1, True, True) 

        >> pred, state = convlstm(x) 
    """ 
    def __init__(self, input_dim, hidden_dim, kernel_size, dec_kernel_size, num_layers,
                batch_first=False, bias=True):
        super().__init__()

        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim 
        self.kernel_size = kernel_size 
        self.dec_kernel_size = dec_kernel_size
        self.num_layers = num_layers 
        self.batch_first = batch_first
        self.bias = bias 
        self.padding = 'same'

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim, 
                                hidden_dim=self.hidden_dim[i],
                                kernel_size=self.kernel_size[i],
                                bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

        self.dec = nn.Conv2d(in_channels=self.hidden_dim[-1]+1,
                            out_channels=1, kernel_size=self.dec_kernel_size, 
                            padding=self.padding, bias=self.bias)

        self.relu = nn.ReLU()
    
    def forward(self, input_tensor, hidden_state=None):
        r"""
        Parameters:
        -----------
        input_tensor: 
            5-D tensor (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state:
            None or list of tuples(h, c)
        
        Returns:
        ----------
        Pred:
            3-D tensor (b, h, w)
        hidden_state:
            None or list of tuples(h, c)
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            # raise NotImplementedError()
            use_random_iter = False
        else:
            use_random_iter = True
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        state_list = []

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                    cur_state=[h, c])
                output_inner.append(h)
            
            if not use_random_iter:
                state_list.append((h, c))
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

        pred = self.relu(self.dec(torch.cat([input_tensor[:, -1, 0, :, :].unsqueeze(dim=1), layer_output[:, -1, :, :, :]], dim=1)))

        if len(state_list) == 0:
            return pred[:, 0, :, :], None
        else:
            return pred[:, 0, :, :], state_list


    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod   # 静态方法无需实例化
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or isinstance(kernel_size, int) or (isinstance(kernel_size, list)
            and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('kernel_size must be int or tuple or list of tuples')
    
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    def summary(self, PATH):
        with open(PATH + "\\log.txt","a") as f:
            f.write("\n-----------MODEL_INFO-----------\n")
            f.write("model:\tConvLSTM\n")
            f.write("hidden_dim:\t" + str(self.hidden_dim) + "\n")
            f.write("kernel_size:\t" + str(self.kernel_size) + "\n")
            f.write("dec_kernel_size:\t" + str(self.dec_kernel_size) + "\n")
            f.write("num_layers:\t" + str(self.num_layers) + "\n")