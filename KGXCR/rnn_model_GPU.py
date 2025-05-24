from sys import int_info
import torch
from torch.autograd import Variable
import data_helpers as dh

#Dream model
class DRModel(torch.nn.Module):
    """
    Input Data: b_1, ... b_i ..., b_t
                b_i stands for user u's ith basket
                b_i = [p_1,..p_j...,p_n]
                p_j stands for the  jth product in user u's ith basket
    """

    def __init__(self, config, item_embeddings, n_rnn_lays, rnn_drops, rnn_lr, device):
        super(DRModel, self).__init__()

        # Model configuration
        self.config = config
        self.item_embeddings = item_embeddings
        self.n_rnn_lays = n_rnn_lays
        self.rnn_drops = rnn_drops
        self.rnn_lr = rnn_lr
        self.device = device

        self.encode = torch.nn.Parameter(self.item_embeddings, requires_grad=True).to(device)  # False if I do not want to update item embeddings
        self.config.embedding_dim = self.item_embeddings.size(1)
        self.pool = {'avg': dh.pool_avg, 'max': dh.pool_max}[config.basket_pool_type]  # Pooling of basket

        # RNN type specify
        if config.rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(torch.nn, config.rnn_type)(input_size=config.embedding_dim,
                                                          hidden_size=config.embedding_dim,
                                                          num_layers=n_rnn_lays,
                                                          batch_first=True,
                                                          dropout=rnn_drops,
                                                          bidirectional=False).to(device)
        else:
            nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[config.rnn_type]
            self.rnn = torch.nn.RNN(input_size=config.embedding_dim,
                                    hidden_size=config.embedding_dim,
                                    num_layers=n_rnn_lays,
                                    nonlinearity=nonlinearity,
                                    batch_first=True,
                                    dropout=rnn_drops,
                                    bidirectional=False).to(device)

    def forward(self, x, lengths, hidden):
        
        ub_seqs = torch.zeros(self.config.batch_size, self.config.seq_len, self.config.embedding_dim, device=self.device)
        for (i, user) in enumerate(x):  # shape of x: [batch_size, seq_len, indices of product]
            #embed_baskets = torch.Tensor(self.config.seq_len, self.config.embedding_dim, device=self.device)
            embed_baskets = torch.zeros(self.config.seq_len, self.config.embedding_dim, device=self.device)

            for (j, basket) in enumerate(user):  # shape of user: [seq_len, indices of product]
                
                basket = torch.tensor(basket, device = self.device).unsqueeze(0)
            
                basket_emb = self.encode[basket]  # [1, len(basket), embedding_dim]
                basket_pooled = self.pool(basket_emb, dim=1).reshape(self.config.embedding_dim)
                embed_baskets[j] = basket_pooled
            # Concat current user's all baskets and append it to users' basket sequence
            ub_seqs[i] = embed_baskets  # shape: [batch_size, seq_len, embedding_dim]

        # Packed sequence as required by pytorch
        packed_ub_seqs = torch.nn.utils.rnn.pack_padded_sequence(ub_seqs, lengths, batch_first=True)

        # RNN
        output, h_u = self.rnn(packed_ub_seqs, hidden)

        # shape: [batch_size, true_len(before padding), embedding_dim]
        dynamic_user, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return dynamic_user, h_u

    def init_weight(self):
        # Init item embedding
        initrange = 0.1
        # self.encode.weight.data.uniform_(-initrange, initrange)
        self.encode.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        # Init hidden states for rnn
        weight = next(self.parameters()).data
        if self.config.rnn_type == 'LSTM':
            return (Variable(weight.new(self.n_rnn_lays, batch_size, self.config.embedding_dim).zero_().to(self.device)),
                    Variable(weight.new(self.n_rnn_lays, batch_size, self.config.embedding_dim).zero_().to(self.device)))
        else:
            #return Variable(torch.zeros(self.n_rnn_lays, batch_size, self.config.embedding_dim))
             return Variable(
                weight.new(self.n_rnn_lays, batch_size, self.config.embedding_dim).zero_().to(self.device)
            )
