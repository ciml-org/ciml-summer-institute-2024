#!/usr/bin/env python3
#
# Train a simple Convolutional Neural Network (CNN) to classify CIFAR images.
#
# https://www.tensorflow.org/tutorials/images/cnn
# https://touren.github.io/2016/05/31/Image-Classification-CIFAR10.html
# https://towardsdatascience.com/deep-learning-with-cifar-10-image-classification-64ab92110d79

import os
import tensorflow as tf
import horovod.tensorflow.keras as hvd

# Initialize Horovod
# https://horovod.readthedocs.io/en/latest/keras.html
# https://horovod.readthedocs.io/en/latest/api.html#module-horovod.tensorflow.keras
hvd.init()

# Pin each GPU to a single MPI process. 
#
# With the typical setup of one GPU per process, set this to local rank. 
# The first process on a node will be allocated the first GPU, the 
# second process will be allocated the second GPU, and so on. 
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# Download training and test image datasets; download only once on each node
# https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data
if hvd.local_rank() == 0:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    hvd.broadcast(0, 0)
else:
    hvd.broadcast(0, 0)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Verify training and ...
assert x_train.shape == (50000, 32, 32, 3)
assert y_train.shape == (50000, 1)
# ... test image dataset sizes
assert x_test.shape == (10000, 32, 32, 3)
assert y_test.shape == (10000, 1)

# Partition training dataset by the number of processes.
examples_per_rank = x_train.shape[0] // hvd.size()
train_begin = examples_per_rank * hvd.rank()
train_end = train_begin + examples_per_rank
x_train = x_train[train_begin:train_end,]
y_train = y_train[train_begin:train_end,]

# Verify training dataset was partitioned.
assert x_train.shape == (examples_per_rank, 32, 32, 3)
assert y_train.shape == (examples_per_rank, 1)

# Normalize the 8-bit (3-channel) RGB image pixel data between 0.0 and 1.0
# https://en.wikipedia.org/wiki/8-bit_color
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define the model and its network architecture. A Sequential model is 
# appropriate for a network with a plain stack of layers, where each 
# layer has exactly one input tensor and one output tensor.
# https://www.tensorflow.org/guide/keras/sequential_model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10),
])

# Print the summary of the model's network architecture
# https://www.tensorflow.org/api_docs/python/tf/keras/Model#summary
if hvd.rank() == 0:
    model.summary()

# Scale the learning rate by the number of processes and wrap the 
# Keras optimizer with Horovod's Distributed Optimizer.
# https://horovod.readthedocs.io/en/latest/keras.html
lr = 0.001 * hvd.size()
opt = tf.keras.optimizers.Adam(learning_rate=lr)
opt = hvd.DistributedOptimizer(opt) 

# Specify an optimizer, a loss function, and metrics, then compile the model.
# Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
# uses hvd.DistributedOptimizer() to compute gradients.
# https://www.tensorflow.org/guide/keras/ain_and_evaluate
# https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#compile
model.compile(
    optimizer=opt,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    experimental_run_tf_function=False,
)

# Horovod: broadcast initial variable states from rank 0 to all other 
# processes. This is necessary to ensure consistent initialization of 
# all workers when training is started with random weights or restored
# from a checkpoint.
hvd_init_bcast = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
]

# Train the model
# https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#fit
model.fit(
    x=x_train, 
    y=y_train,
    batch_size=256, 
    epochs=50, 
    validation_split=0.2, 
    verbose=2 if hvd.rank() == 0 else 0,
    callbacks=hvd_init_bcast,
)

# Evaluate the model and its accuracy
# https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#evaluate
if hvd.rank() == 0:
    model.evaluate(
        x=x_test, 
        y=y_test,
        batch_size=256,
        verbose=2,
    )

# Save the model
if hvd.rank() == 0:
    model.save('saved_model.o'+os.environ['SLURM_JOB_ID'])
