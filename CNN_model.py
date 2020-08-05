from keras import layers, models

class CNN(object):
    def __init__(self, num_filter, kernel_size):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(num_filter, (kernel_size, kernel_size), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(num_filter * 2, (kernel_size, kernel_size), activation='relu'))
        self.model.add(layers.MaxPooling2D((kernel_size, kernel_size)))
        self.model.add(layers.Conv2D(num_filter * 2, (kernel_size, kernel_size), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10, activation='softmax'))
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
        self.model.summary()
    
    def fit(self, x, y, batch_size=64, epochs=60, path=None):
        hist = self.model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
        self.model.save(path + f"CNN.h5")
        return hist

    def evaluate(self, x, y):
        test_loss, test_acc = self.model.evaluate(x, y)
        return test_loss, test_acc