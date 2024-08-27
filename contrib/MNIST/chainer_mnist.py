#!/usr/bin/env python3

import argparse
import chainer
import os
import numpy as np
from chainer import optimizers
from chainer.iterators import SerialIterator
from chainer.training import StandardUpdater, Trainer
from chainer.training.extensions import LogReport, PrintReport, Evaluator
import chainer.functions as F
import chainer.links as L
from chainer.utils import type_check

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        _ = f.read(16)
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(-1, 784)
        data = data.astype(np.float32) / 255.0
    return data

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        _ = f.read(8)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        labels = labels.astype(np.int32)  # Ensure labels are int32
    return labels

class Net(chainer.Chain):
    def __init__(self):
        super(Net, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 32, 3, 1)
            self.conv2 = L.Convolution2D(32, 64, 3, 1)
            self.fc1 = L.Linear(9216, 128)
            self.fc2 = L.Linear(128, 10)

    def __call__(self, x):
        x = F.reshape(x, (x.shape[0], 1, 28, 28))
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2)
        h = F.dropout(h, 0.25)
        h = F.reshape(h, (len(x), -1))
        h = F.relu(self.fc1(h))
        h = F.dropout(h, 0.5)
        return F.softmax(self.fc2(h))

def main():
    parser = argparse.ArgumentParser(description='Chainer MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='learning rate step gamma (default: 0.7)')
    parser.add_argument('--gpu', type=int, default=0,  # Set default GPU ID to 0
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='for Saving the current Model')
    args = parser.parse_args()

    chainer.config.autotune = True

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        chainer.cuda.check_cuda_available()

    chainer.backends.cuda.set_max_workspace_size(512 * 1024 * 1024)

    np.random.seed(args.seed)

    data_dir = 'data/MNIST/raw'

    train_images = load_mnist_images(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    train_labels = load_mnist_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte'))

    test_images = load_mnist_images(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    test_labels = load_mnist_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))

    train = chainer.datasets.TupleDataset(train_images, train_labels)
    test = chainer.datasets.TupleDataset(test_images, test_labels)

    train_iter = SerialIterator(train, args.batch_size, shuffle=True)
    test_iter = SerialIterator(test, args.test_batch_size, repeat=False, shuffle=False)

    model = L.Classifier(Net())
    if args.gpu >= 0:
        model.to_gpu()

    optimizer = optimizers.AdaDelta(args.lr)
    optimizer.setup(model)

    updater = StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = Trainer(updater, (args.epochs, 'epoch'), out='result')

    trainer.extend(Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(LogReport())
    trainer.extend(PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
    trainer.run()

    if args.save_model:
        chainer.serializers.save_npz('mnist_chainer.npz', model)

if __name__ == '__main__':
    main()