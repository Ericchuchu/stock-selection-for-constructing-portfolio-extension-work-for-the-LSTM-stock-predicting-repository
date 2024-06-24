import torch
import torch.nn as nn
from torch.autograd import Variable

class ResGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, residual_on = True, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(ResGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.residual_on = residual_on
        self.weight_ir = nn.Parameter(torch.rand(hidden_size, input_size)).to(device)
        self.weight_hr = nn.Parameter(torch.rand(hidden_size, hidden_size)).to(device)
        self.bias_ir = nn.Parameter(torch.rand(hidden_size)).to(device)
        self.bias_hr = nn.Parameter(torch.rand(hidden_size)).to(device)
        self.weight_in = nn.Parameter(torch.rand(hidden_size, input_size)).to(device)
        self.weight_hn = nn.Parameter(torch.rand(hidden_size, hidden_size)).to(device)
        self.bias_in = nn.Parameter(torch.rand(hidden_size)).to(device)
        self.bias_hn = nn.Parameter(torch.rand(hidden_size)).to(device)
        self.weight_ri = nn.Parameter(torch.rand(hidden_size, input_size)).to(device)  # For residual connection

    def forward(self, x, h):
        r_gate = torch.sigmoid(torch.mm(x, self.weight_ir.t()) + self.bias_ir + torch.mm(h, self.weight_hr.t()) + self.bias_hr)
        z_gate = torch.sigmoid(torch.mm(x, self.weight_ir.t()) + self.bias_ir + torch.mm(h, self.weight_hr.t()) + self.bias_hr)
        n_tilde = torch.tanh(torch.mm(x, self.weight_in.t()) + self.bias_in + r_gate * (torch.mm(h, self.weight_hn.t()) + self.bias_hn))
        h_next = (1 - z_gate) * n_tilde + z_gate * h

        # Residual connection
        if self.residual_on:
            if self.input_size == self.hidden_size:
                h_next = h_next + x
            else:
                h_next = h_next + torch.mm(x, self.weight_ri.t())
            return h_next
        else:
            return h_next
    
    def init_hidden(self, batch_size, dim, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        return (Variable(torch.zeros(batch_size, dim)).to(device))

class ResGRU(nn.Module):
    def __init__(self, input_size, hidden_size, out_dim=1, step=5, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(ResGRU, self).__init__()
        self.input_size = [input_size] + hidden_size
        self.hidden_size = hidden_size
        self.num_layers = len(hidden_size)
        self.step = step
        self.device = device
        self._all_layers = []
        self.dropout_layers = nn.ModuleList([nn.Dropout(0.01) for hs in hidden_size])  
        self.linear = nn.Linear(in_features=hidden_size[-1], out_features=out_dim)

        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            if i == 0:
                cell = ResGRUCell(self.input_size[i], self.hidden_size[i], device=device)
            if i > 0:
                cell = ResGRUCell(self.input_size[i], self.hidden_size[i], residual_on=False, device=device)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        for step in range(self.step):
            x = input[:, step, :]
            for i in range(self.num_layers):
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _ = x.size()
                    h = getattr(self, name).init_hidden(batch_size=bsize, dim=self.hidden_size[i], device=self.device)
                    internal_state.append(h)

                h = internal_state[i]
                x = getattr(self, name)(x, h)
                x = self.dropout_layers[i](x)
                internal_state[i] = x

            outputs = x
            outputs = self.linear(outputs).squeeze()

        return outputs
    
# test
if __name__ == '__main__':
    x = torch.rand(1, 5, 8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResGRU(input_size=8, hidden_size=[256, 128, 64], out_dim = 1, step = 5, device = device)
    print(model(x).shape)