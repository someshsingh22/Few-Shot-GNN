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


args = argparse.Namespace(
    folder="./data",
    batch_size=300,
    batch_size_test=300,
    test_iterations=50,
    dataset="omniglot",
    dataset_root="datasets",
    dec_lr=10000,
    decay_interval=10000,
    iterations=100000,
    log_interval=50,
    test_interval=100,
    lr=0.001,
    no_cuda=False,
    save_interval=30000,
    seed=22,
    test_N_shots=1,
    test_N_way=5,
    train_N_shots=1,
    train_N_way=5,
    unlabeled_extra=0,
)

train_dataset, test_dataset = omniglot(
    folder=Path(args.folder),
    shots=args.train_N_shots,
    ways=args.train_N_way,
    shuffle=False,
    test_shots=args.test_N_shots,
    meta_split="train",
    download=True,
), omniglot(
    folder=Path(args.folder),
    shots=args.train_N_shots,
    ways=args.train_N_way,
    shuffle=False,
    test_shots=args.test_N_shots,
    meta_split="test",
    download=True,
)


def load_batch_inputs(self, batch, device):
    batches_xi, labels_yi = batch["train"]
    batches_xi, labels_yi = [
        batches_xi[:, i].to(device=device) for i in range(args.train_N_way)
    ], [F.one_hot(labels_yi[:, i]).to(device=device) for i in range(args.train_N_way)]
    idx = random.randint(0, args.train_N_way - 1)
    batch_x, label_x = batch["test"]
    batch_x, label_x = batch_x[:, idx], F.one_hot(label_x[:, idx])
    return batch_x.to(device=device), label_x.to(device=device), batches_xi, labels_yi


BatchMetaDataLoader.load_batch_inputs = load_batch_inputs
train_loader, test_loader = BatchMetaDataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1
), BatchMetaDataLoader(
    test_dataset, batch_size=args.batch_size_test, shuffle=True, num_workers=1
)
test_iterator = iter(test_loader)

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
    device = "cuda"
else:
    logging.info("Using CPU")
    device = "cpu"
np.random.seed(args.seed)
random.seed(args.seed)

model = FSGNN(args, emb_size=64)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.MultiplicativeLR(
    optimizer, lr_lambda=lambda iter: 0.5 ** (iter / args.dec_lr)
)
model.train()

counter, total_loss = 0, 0
val_acc, val_acc_aux, test_acc = 0, 0, 0

for batch_idx, batch in enumerate(train_loader):
    if counter >= args.iterations:
        break
    batch_x, label_x, batches_xi, labels_yi = train_loader.load_batch_inputs(
        batch, args.device
    )
    optimizer.zero_grad()
    _, loss = model(batch_x, label_x, batches_xi, labels_yi)
    loss.backward()
    optimizer.step()
    scheduler.step()
    total_loss += loss.item()
    counter += 1
    if (batch_idx + 1) % args.log_interval == 0:
        print(
            f"Train Iter: {batch_idx+1} \t\tLoss_d_metric: {total_loss / counter:.6f}"
        )
        counter, total_loss = 0, 0

    if (batch_idx + 1) % args.test_interval == 0:
        iteration, test_loss, test_correct, test_total = 0, 0, 0, 0
        with torch.no_grad():
            while iteration < args.test_iterations:
                test_batch = test_iterator.next()
                batch_x, label_x, batches_xi, labels_yi = test_loader.load_batch_inputs(
                    test_batch, args.device
                )
                test_total += label_x.size()[0]
                logsoft_prob, loss = model(batch_x, label_x, batches_xi, labels_yi)
                test_loss += loss.item()
                true = label_x.argmax(dim=1).cpu().numpy()
                predictions = logsoft_prob.argmax(dim=1).cpu().numpy()
                test_correct += (true == predictions).sum()
                iteration += 1
            print(
                f"Test Iter: {batch_idx+1} \t\tAccuracy {test_correct / test_total:.3f}"
            )
