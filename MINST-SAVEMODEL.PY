

#将训练的参数进行保存，方式有：1.创建检查点文件，使用checkpoint_path进行检查点文件定义，在fit中使用callback方法
#                            2.创建周期检查点文件，在新模型上使用时加载检查点文件中的训练结果
#                            3.使用手动保存权值的方法，model.save_weight（路径）
#  要在新模型上使用已经训练好的参数，只需要使用model.load_weights()方法
from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
from tensorflow import keras


(train_images, train_labels),(test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
#reshape方法是将数组的形状进行改变，reshape中的参数就是数组需要改变的形状
#比如当reshape（2， 3）是将数组转变成2D，方式为两个数组，每个数组中由三个元素组成
#而当reshape（-1，value）这种情况下是让numpy自动转换维度，根据value的值自动计算维度
train_images = train_images[:1000].reshape(-1, 28*28)/255.0
test_images = test_images[:1000].reshape(-1, 28*28)/255.0


def create_model():
      model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784, )),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
      ])

      model.compile(optimizer=tf.keras.optimizers.Adam(),
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      metrics=['accuracy']
      )

      return model

# checkpoint_path = 'training_1/cp.ckpt'
# checkpoint_dir = os.path.dirname(checkpoint_path)
# cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
#                                                 save_weights_only=True,
#                                                 verbose=1)

# model = create_model()
# model.fit(train_images, train_labels, epochs=10,
#           validation_data=(test_images, test_labels),
#           callbacks=[cp_callback])

# model.load_weights(checkpoint_path)
# loss, acc = model.evaluate(test_images, test_labels)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     checkpoint_path, verbose=1, save_weights_only=True,
#     period=5
# )

model = create_model()
model.fit(train_images, train_labels,
          epochs=50, callbacks=[cp_callback],
          validation_data=(test_images,test_labels),
          verbose=0)
model.save('my_model.h5')
new_model = keras.models.load_model('my_model.h5')
new_model.summary()
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
# latest = tf.train.latest_checkpoint(checkpoint_dir)
# latest
# model = create_model()
# model.load_weights(latest)
# loss, acc = model.evaluate(test_images, test_labels)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# model.save_weights('./checkpoints/my_checkpoint')

# model = create_model()
# model.load_weights('./checkpoints/my_checkpoint')

# loss, acc = model.evaluate(test_images, test_labels)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))




