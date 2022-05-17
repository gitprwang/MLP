import torch
import torch.nn as nn
import torch.nn.functional as F
import copy 
def clones(module, N):
    '''
    Produce N identical layers.
    :param module: nn.Module
    :param N: int
    :return: torch.nn.ModuleList
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SublayerConnection(nn.Module):
    '''
    A residual connection followed by a layer norm
    '''
    def __init__(self, size, dropout, residual_connection, use_LayerNorm):
        super(SublayerConnection, self).__init__()
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        self.dropout = nn.Dropout(dropout)
        if self.use_LayerNorm:
            self.norm = nn.LayerNorm(size)

    def forward(self, x, sublayer):
        '''
        :param x: (batch, N, T, d_model)
        :param sublayer: nn.Module
        :return: (batch, N, T, d_model)
        '''
        if self.residual_connection and self.use_LayerNorm:
            return x + self.dropout(sublayer(self.norm(x)))
        if self.residual_connection and (not self.use_LayerNorm):
            return x + self.dropout(sublayer(x))
        if (not self.residual_connection) and self.use_LayerNorm:
            return self.dropout(sublayer(self.norm(x)))

class GraphMLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_nodes=307, act=torch.relu):
        super(GraphMLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim 
        self.act = act
        self.w = nn.Parameter(torch.randn(in_dim, out_dim), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(out_dim), requires_grad=True)
        self.ln = nn.LayerNorm([num_nodes, out_dim])

    def forward(self, x):
        # x.shape : B,N,T,d
        x = x.transpose(1,2)
        x = torch.einsum('abcd,de->abce', x, self.w)
        x = x + self.b
        x = self.act(x)
        x = self.ln(x)
        return x.transpose(1,2)

class TemporalMLP(nn.Module):
    def __init__(self, in_dim, out_dim, in_len, out_len, act=torch.relu):
        super(TemporalMLP, self).__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.w0 = nn.Parameter(torch.randn(in_len, out_len), requires_grad=True)
        self.w = nn.Parameter(torch.randn(in_dim, out_dim), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(out_dim), requires_grad=True)
        self.act = act

    def forward(self, x):
        # x.shape : B,N,T,d
        x = torch.einsum('abcd,de->abce', x, self.w)
        x = x + self.b
        x = self.act(x) if self.act is not None else x
        x = torch.einsum('abcd,ce->abed', x, self.w0)
        # x = x+self.b
        return x

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, gcn, dropout, residual_connection=True, use_LayerNorm=True):
        super(EncoderLayer, self).__init__()
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        self.self_attn = self_attn
        self.feed_forward_gcn = gcn
        if residual_connection or use_LayerNorm:
            self.sublayer = clones(SublayerConnection(size, dropout, residual_connection, use_LayerNorm), 2)
        self.size = size

    def forward(self, x):
        '''
        :param x: src: (batch_size, N, T_in, F_in)
        :return: (batch_size, N, T_in, F_in)
        '''
        if self.residual_connection or self.use_LayerNorm:
            x = self.sublayer[0](x, lambda x: self.self_attn(x))
            return self.sublayer[1](x, self.feed_forward_gcn)
        else:
            x = self.self_attn(x)
            return self.feed_forward_gcn(x)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        '''
        :param layer:  EncoderLayer
        :param N:  int, number of EncoderLayers
        '''
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x):
        '''
        :param x: src: (batch_size, N, T_in, F_in)
        :return: (batch_size, N, T_in, F_in)
        '''
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class MLPASTGNN(nn.Module):
    def __init__(self, in_dim, out_dim, in_len, out_len, num_nodes, num_layers=4, dropout=.0, d_model=64, residual_connection=True, use_LayerNorm=True):
        super(MLPASTGNN, self).__init__()
        c = copy.deepcopy
        self.in_dim = in_dim
        self.d_model = d_model
        self.src_dense = nn.Linear(in_dim, d_model)
        self.gcn = GraphMLP(d_model, d_model, num_nodes=num_nodes)
        self.attn_ss = TemporalMLP(d_model, d_model, in_len, out_len)
        self.generator = nn.Linear(d_model, out_dim)
        encoderLayer = EncoderLayer(d_model, self.attn_ss, c(self.gcn), dropout, residual_connection=residual_connection, use_LayerNorm=use_LayerNorm)
        self.encoder = Encoder(encoderLayer, num_layers)
        decoderLayer = EncoderLayer(d_model, self.attn_ss, c(self.gcn), dropout, residual_connection=residual_connection, use_LayerNorm=use_LayerNorm)
        self.decoder = Encoder(decoderLayer, num_layers)
    def forward(self, x):
        # x:B,T,N,d
        x = x.transpose(1,2)
        x = self.src_dense(x)
        h = self.encoder(x)
        out = self.generator(self.decoder(h))
        out = out.transpose(1,2) 
        return out



