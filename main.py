from keras.datasets import mnist
from keras.utils import to_categorical
from CNN_model import CNN
import os
import time
import matplotlib.pyplot as plt

model_path = ["model/loss/", "model/accuracy/", "model/weight/"]
num_filter = 32
kernel_size = 3

def main():
    (train_images , train_labels) , (test_images , test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28 , 28, 1))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28 , 28, 1))
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    model = CNN(num_filter, kernel_size)
    hist = model.fit(train_images, train_labels, epochs=5, batch_size=32, path=model_path[2])

    history_dict = hist.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1,len(loss_values) + 1)
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    drawImage(epochs, loss_values, val_loss_values, 'Epochs', 'Loss', 'Training and validation loss', model_path[0] + 'loss.png', 'loss')
    drawImage(epochs, acc, val_acc, 'Epochs', 'Accuracy', 'Training and validation accuracy', model_path[1] + 'accuracy.png', 'acc')

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print(f'Accuracy : {round(test_acc * 100)}%')

def drawImage(x, y1, y2, xlabel, ylabel, title, path, mode):
    plt.plot(x, y1, 'b', label=f'Training {mode}')
    plt.plot(x, y2, 'r', label=f'Validation {mode}')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()
    plt.legend()
    plt.savefig(path)
    plt.clf()

def createFile(path):
    if not os.path.isdir(path):
        os.makedirs(path)

if __name__ == "__main__":
    for path in model_path:
        createFile(path)

    startTime = time.time()
    main()
    endTime = time.time()
    print(f"Time taken : {round(endTime - startTime)}s")