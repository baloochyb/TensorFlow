import tensorflow as tf
# from tensorflow.python.keras import backend as k

# print(tf.keras.__version__)
# const = k.constant([[42,24],[11,99]], dtype=tf.float16, shape=[2,2])
# print(const)
# Prepare Data
mnist = tf.keras.datasets.mnist
(train_x,train_y), (test_x, test_y) = mnist.load_data()
epochs=10
batch_size = 32 # 32 is default in fit method but specify anyway
train_x, test_x = tf.cast(train_x/255.0, tf.float32), tf.cast(test_x/255.0, tf.float32)
train_y, test_y = tf.cast(train_y,tf.int64), tf.cast(test_y,tf.int64)
# Build Model
model1 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model1.summary()
# Compile Model
optimiser = tf.keras.optimizers.Adam()
model1.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Fit Model
model1.fit(train_x, train_y, batch_size=batch_size, epochs=epochs)
# Evaluate Mosel
model1.evaluate(test_x, test_y)

# An alternative method:

model2 = tf.keras.models.Sequential()
model2.add(tf.keras.layers.Flatten())
model2.add(tf.keras.layers.Dense(512, activation='relu'))
model2.add(tf.keras.layers.Dropout(0.2))
model2.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

model2.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

model2.fit(train_x, train_y, batch_size=batch_size, epochs=epochs)

model2.evaluate(test_x, test_y)
