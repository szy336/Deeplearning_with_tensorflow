import tensorflow as tf
from tensorflow import keras
import numpy as np
# 获取imdb的语料，这个语料分为训练集和测试集，并且已经将单词转化为了词在字典中的数进行表达。
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data,test_labels) = imdb.load_data(num_words=10000)
# 将数组化的句转化为英文单词
word_index = imdb.get_word_index()

word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key)
                           for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# print(decode_review(train_data[0]))

# 将训练集和测试集的数据进行归一化，使其有着相同的长度
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data, value=word_index["<PAD>"], padding='post', maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(
    train_data, value=word_index["<PAD>"], padding='post', maxlen=256)
print(len(train_data[0]), len(train_data[1]))
print(test_data[0])



vocab_size = 10000 #定义一个数据长度，用来表示训练数据的长度

#定义神经网络结构，使用keras.Sequential()方法
model = keras.Sequential()
#调用add方法添加网络层，Embedding层用来进行词嵌入操作，第一个参数表示数据量的总数
#这里是10000个单词，16代表output_dim，即输出张量的维度，和全连接层Dense的维度一致
#这里model.output_shape == (10000, 16, none) 2D-3D
model.add(keras.layers.Embedding(vocab_size, 16))
#GlobalAveragePooling1D层是池化层，这里使用1D信号进行最大池化  3D-3D
model.add(keras.layers.GlobalAveragePooling1D())
#Dense是全连接层，这里使用activation激活函数是relu，在Dense中的input_shape = （value， ）
#这里表示张量的是由（*， value）构成的张量
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()
#使用complie方法对学习过程进行配置，接收三个参数
#optimizer优化器参数，使用什么反向传播算法
#loss损失函数，模型最小化的目标函数
#metrics指定列表，？
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy', metrics=['accuracy'])

x_val = train_data[:10000]
partial_x_train = test_data[10000:]

y_val = train_labels[:10000]
partial_y_train = test_labels[10000:]
#使用model.fit方法让模型开始训练
history = model.fit(partial_x_train,
                    partial_y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val),
                    verbose=1)
results = model.evaluate(test_data, test_labels)
print(results)

history_dict = history.history
history_dict.keys()
# dict_keys(['loss','val_loss','val_acc','acc'])

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

