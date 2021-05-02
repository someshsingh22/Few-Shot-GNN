from __future__ import print_function
import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import random

from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader

from Layers import OmniglotEmbedding, MetricNN

parser = argparse.ArgumentParser(
    description="Few-Shot Learning with Graph Neural Networks"
)
parser.add_argument(
    "--folder", type=str, default="./data", metavar="folder", help="Data Folder"
)

parser.add_argument(
    "--batch_size", type=int, default=10, metavar="batch_size", help="Size of batch)"
)
parser.add_argument(
    "--batch_size_test",
    type=int,
    default=10,
    metavar="batch_size",
    help="Size of batch)",
)
parser.add_argument(
    "--iterations",
    type=int,
    default=50000,
    metavar="N",
    help="number of epochs to train ",
)
parser.add_argument(
    "--decay_interval",
    type=int,
    default=10000,
    metavar="N",
    help="Learning rate decay interval",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.001,
    metavar="LR",
    help="learning rate (default: 0.01)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.5,
    metavar="M",
    help="SGD momentum (default: 0.5)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="enables CUDA training"
)
parser.add_argument(
    "--seed", type=int, default=22, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=20,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--save_interval",
    type=int,
    default=300000,
    metavar="N",
    help="how many batches between each model saving",
)
parser.add_argument(
    "--test_interval",
    type=int,
    default=2000,
    metavar="N",
    help="how many batches between each test",
)
parser.add_argument(
    "--test_N_way",
    type=int,
    default=5,
    metavar="N",
    help="Number of classes for doing each classification run",
)
parser.add_argument(
    "--train_N_way",
    type=int,
    default=5,
    metavar="N",
    help="Number of classes for doing each training comparison",
)
parser.add_argument(
    "--test_N_shots", type=int, default=1, metavar="N", help="Number of shots in test"
)
parser.add_argument(
    "--train_N_shots",
    type=int,
    default=1,
    metavar="N",
    help="Number of shots when training",
)
parser.add_argument(
    "--dec_lr",
    type=int,
    default=10000,
    metavar="N",
    help="Decreasing the learning rate every x iterations",
)
args = parser.parse_args()

train_dataset = omniglot(
    folder=Path(args.folder),
    shots=args.train_N_shots,
    ways=args.train_N_way,
    shuffle=False,
    test_shots=args.test_N_shots,
    meta_split="train",
    download=True,
)

test_dataset = omniglot(
    folder=Path(args.folder),
    shots=args.train_N_shots,
    ways=args.train_N_way,
    shuffle=False,
    test_shots=args.test_N_shots,
    meta_split="train",
    download=True,
)


def adjust_learning_rate(optimizers, lr, iter):
    new_lr = lr * (0.5**(int(iter/args.dec_lr)))

    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


train_loader = BatchMetaDataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1
)
test_loader = BatchMetaDataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1
)

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    logging.info(
        "Using GPU : "
        + str(torch.cuda.current_device())
        + " from "
        + str(torch.cuda.device_count())
        + " devices"
    )
    torch.cuda.manual_seed(args.seed)
else:
    logging.info("Using CPU")

# Try to load models
Embedder = OmniglotEmbedding(args, 64)
metric_nn = MetricNN(args, 64)
# softmax_module = models.SoftmaxModule()

if args.cuda:
    Embedder.cuda()
    metric_nn.cuda()

weight_decay = 0
opt_Embedder = optim.Adam(Embedder.parameters(), lr=args.lr, weight_decay=weight_decay)
opt_metric_nn = optim.Adam(
    metric_nn.parameters(), lr=args.lr, weight_decay=weight_decay
)

Embedder.train()
metric_nn.train()
counter = 0
total_loss = 0
val_acc, val_acc_aux = 0, 0
test_acc = 0

for batch_idx, batch in enumerate(train_loader):
    batches_xi, labels_yi = batch["train"]
    batches_xi, labels_yi = [batches_xi[:, i] for i in range(args.train_N_way)], [
        F.one_hot(labels_yi[:, i]) for i in range(args.train_N_way)
    ]
    idx = random.randint(0, args.train_N_way - 1)
    batch_x, label_x = batch["test"]
    batch_x, label_x = batch_x[:, idx], F.one_hot(label_x[:, idx])

    z = Embedder(batch_x)[-1]
    zi_s = [Embedder(batch_xi)[-1] for batch_xi in batches_xi]

    # Compute metric from embeddings
    out_metric, out_logits = metric_nn(z, zi_s, labels_yi)
    logsoft_prob = F.softmax(out_logits)

    # Loss
    label_x_numpy = label_x.cpu().data.numpy()
    formatted_label_x = np.argmax(label_x_numpy, axis=1)
    formatted_label_x = Variable(torch.LongTensor(formatted_label_x))
    if args.cuda:
        formatted_label_x = formatted_label_x.cuda()
    loss = F.nll_loss(logsoft_prob, formatted_label_x)
    loss.backward()

    opt_Embedder.step()
    opt_metric_nn.step()
    adjust_learning_rate(
        optimizers=[opt_Embedder, opt_metric_nn], lr=args.lr, iter=batch_idx
    )
