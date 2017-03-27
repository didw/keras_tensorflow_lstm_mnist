from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os, shutil
from keras import backend as K
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import TensorBoard
import time

class KerasLSTM():
    def __init__(self, h_size=128, n_inputs=28, n_steps=28, n_classes=10, l_r=0.001):
        # parameters init
        l_r = l_r
        self.n_inputs = n_inputs
        self.n_steps = n_steps
        n_classes = n_classes
        self.model_dir = 'model/keras/lstm'

        ## build graph
        #tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        K.set_session(tf.Session(config=config))

        self.model = Sequential()
        self.model.add(LSTM(h_size, input_shape=(n_steps, n_inputs)))
        self.model.add(Dense(n_classes, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

        if os.path.exists('/tmp/keras'): shutil.rmtree('/tmp/keras')
        self.tb = TensorBoard(log_dir='/tmp/keras/MNIST', histogram_freq=0, write_graph=True, write_images=False)

    def fit(self, X_data, Y_data, n_epoch=10, batch_size=128):
        self.model.fit(X_data, Y_data, epochs=n_epoch, batch_size=batch_size, verbose=1, callbacks=[self.tb])

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # saving model
        json_model = self.model.to_json()
        open('%s/model.json'%self.model_dir, 'w').write(json_model)
        self.model.save_weights('%s/model.h5'%self.model_dir, overwrite=True)
        print("Model saved in %s" % self.model_dir)

    def predict(self, X_test):
        estimator = model_from_json(open('%s/model.json'%self.model_dir).read())
        estimator.load_weights('%s/model.h5'%self.model_dir)
        return estimator.predict(X_test)

def main():
    #load mnist data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    keras_lstm = KerasLSTM()
    t1 = time.time()
    keras_lstm.fit(mnist.train.images.reshape(-1, 28, 28), mnist.train.labels)
    t2 = time.time()
    print('training time: %s' % (t2-t1))
    pred = keras_lstm.predict(mnist.test.images.reshape(-1, 28, 28))
    t3 = time.time()
    print('predict time: %s' % (t3-t2))
    test_lab = mnist.test.labels
    print("accuracy: ", np.mean(np.equal(np.argmax(pred,1), np.argmax(test_lab,1)))*100)

if __name__ == '__main__':
    main()
