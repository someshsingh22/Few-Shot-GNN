import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from .GNN import GNN_nl_omniglot


class MetricNN(nn.Module):
    def __init__(self, args, emb_size):
        super(MetricNN, self).__init__()
        self.emb_size = emb_size
        self.args = args
        num_inputs = self.emb_size + self.args.train_N_way
        self.gnn_obj = GNN_nl_omniglot(args, num_inputs, nf=96, J=1)

    def forward(self, z, zi_s, labels_yi):
        zero_pad = Variable(torch.zeros(labels_yi[0].size()))
        if self.args.cuda:
            zero_pad = zero_pad.cuda()

        labels_yi = [zero_pad] + labels_yi
        zi_s = [z] + zi_s

        nodes = [torch.cat([zi, label_yi], 1) for zi, label_yi in zip(zi_s, labels_yi)]
        nodes = [node.unsqueeze(1) for node in nodes]
        nodes = torch.cat(nodes, 1)

        logits = self.gnn_obj(nodes).squeeze(-1)
        outputs = F.sigmoid(logits)

        return outputs, logits
