import tensorflow as tf
from tensorflow.python.keras.models import load_model
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

# Build Model (First Method) --------------------------------------------------------

model1 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
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

# An alternative method --------------------------------------------------------------

model2 = tf.keras.models.Sequential()
model2.add(tf.keras.layers.Flatten())
model2.add(tf.keras.layers.Dense(512, activation='relu'))
model2.add(tf.keras.layers.Dropout(0.2))
model2.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

model2.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

model2.fit(train_x, train_y, batch_size=batch_size, epochs=epochs)

model2.evaluate(test_x, test_y)

# An alternative method --------------------------------------------------------------

inputs = tf.keras.Input(shape=(28,28)) # Returns a 'placeholder' tensor
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(512, activation='relu',name='d1')(x)
x = tf.keras.layers.Dropout(0.2)(x)
predictions = tf.keras.layers.Dense(10,activation=tf.nn.softmax, name='d2')(x)
model3 = tf.keras.Model(inputs=inputs, outputs=predictions)

# Subclassing the Keras Model ---------------------------------------------------------

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
    # Define your layers here.
        inputs = tf.keras.Input(shape=(28,28)) # Returns a placeholder tensor
        self.x0 = tf.keras.layers.Flatten()
        self.x1 = tf.keras.layers.Dense(512, activation='relu',name='d1')
        self.x2 = tf.keras.layers.Dropout(0.2)
        self.predictions = tf.keras.layers.Dense(10,activation=tf.nn.softmax, name='d2')
    def call(self, inputs):
    # This is where to define your forward pass
    # using the layers previously defined in `__init__`
        x = self.x0(inputs)
        x = self.x1(x)
        x = self.x2(x)
        return self.predictions(x)

model4 = MyModel()

model4 = MyModel()
batch_size = 32
steps_per_epoch = len(train_x.numpy())//batch_size
print(steps_per_epoch)
model4.compile (optimizer= tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
model4.fit(train_x, train_y, batch_size=batch_size, epochs=epochs)
model4.evaluate(test_x, test_y)

# data pipelines -----------------------------------------------------------------------

batch_size = 32
buffer_size = 10000
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(batch_size).shuffle(buffer_size)
train_dataset = train_dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
train_dataset = train_dataset.repeat()

test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(batch_size).shuffle(buffer_size)
test_dataset = train_dataset.repeat()

steps_per_epoch = len(train_x)//batch_size # required because of the repeat on the dataset
optimiser = tf.keras.optimizers.Adam()
model4.compile (optimizer= optimiser, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
model4.fit(train_dataset, batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch)

# Saving and loading Keras models

model4.save('./model_name.h5')
new_model = load_model('./model_name.h5')

model3.save_weights('./model_weights.h5')
model3.load_weights('./model_weights.h5')
