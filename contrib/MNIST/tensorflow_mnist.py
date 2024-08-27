#!/usr/bin/env python3

import os
import argparse
import numpy as np
import tensorflow as tf
from keras import datasets, layers, models

class Net(tf.keras.Model):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = layers.Conv2D(32, 3, kernal_size=(3,3) ,activation='relu', input_shape=(32, 32, 3))
        self.conv2 = layers.Conv2D(64, 3, activation='relu')
        self.pool = layers.MaxPooling2D((2, 2))
        self.dropout1 = layers.Dropout(0.25)
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(128, activation='relu')
        self.dropout2 = layers.Dropout(0.5)
        self.fc2 = layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        return self.fc2(x)


def train(args, model, train_dataset, optimizer, epoch):
    for step, (data, target) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            output = model(data, training=True)
            loss = tf.keras.losses.SparseCategoricalCrossentropy()(target, output)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if step % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(data), len(train_dataset) * args.batch_size,
                100. * step / len(train_dataset), loss.numpy()))

def test(args, model, test_dataset):
    test_loss = 0
    correct = 0
    total_samples = 0

    for data, target in test_dataset:
        output = model(data, training=False)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)(target, output)
        test_loss += loss.numpy()
        pred = np.argmax(output.numpy(), axis=1)
        correct += np.sum(pred == target.numpy())
        total_samples += len(data)

    test_loss /= (len(test_dataset) * 100)
    accuracy = (correct / total_samples) * 100

    with open('output/tensorflow_keras_results.txt', 'a') as file:
        file.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, total_samples, accuracy))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='TensorFlow MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='for saving the current model')
    args = parser.parse_args()

    if args.cuda:
        device = '/GPU:0'
    else:
        device = '/CPU:0'

    with tf.device(device):
        # Load MNIST data
    
        (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train, x_test = x_train[..., np.newaxis], x_test[..., np.newaxis]

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(args.batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(args.test_batch_size)

        model = Net()

        optimizer = tf.keras.optimizers.Adadelta(learning_rate=args.lr)

        for epoch in range(1, args.epochs + 1):
            train(args, model, train_dataset, optimizer, epoch)
            test(args, model, test_dataset)

        if args.save_model:
            model.save('mnist_tf_model')


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tensorflow warnings
    main()
