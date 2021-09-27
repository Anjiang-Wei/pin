# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1kP4GR9jWOjY-Y8nX_-dThZgD7uewIJGN

##### Copyright 2019 The TensorFlow Authors.
"""

#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""# 训练后整数量化


## 概述

整数量化是一种优化策略，可将 32 位浮点数（如权重和激活输出）转换为 8 位定点数。这样可以缩减模型大小并加快推理速度，这对低功耗设备（如[微控制器](https://tensorflow.google.cn/lite/microcontrollers)）很有价值。仅支持整数的加速器（如 [Edge TPU](https://coral.ai/)）也需要使用此数据格式。

在本教程中，您将从头开始训练一个 MNIST 模型、将其转换为 TensorFlow Lite 文件，并使用[训练后量化](https://tensorflow.google.cn/lite/performance/post_training_quantization)对其进行量化。最后，您将检查转换后模型的准确率并将其与原始浮点模型进行比较。

实际上，对模型进行量化的程度有几种选项。在本教程中，您将执行“全整数量化”，它会将所有权重和激活输出转换为 8 位整数数据，而其他策略可能会将部分数据保留为浮点。

要详细了解各种量化策略，请阅读 [TensorFlow Lite 模型优化](https://tensorflow.google.cn/lite/performance/model_optimization)。

## 设置

为了量化输入和输出张量，我们需要使用 TensorFlow r2.3 中新添加的 API：
"""

import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
import numpy as np
assert float(tf.__version__[:3]) >= 2.3

"""## 生成 TensorFlow 模型

我们将构建一个简单的模型来对 [MNIST 数据集](https://tensorflow.google.cn/datasets/catalog/mnist)中的数字进行分类。

此训练不会花很长时间，因为只对模型进行 5 个周期的训练，训练到约 98% 的准确率。
"""

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

# Define the model architecture
model = tf.keras.Sequential([
  tf.keras.layers.InputLayer(input_shape=(28, 28)),
  tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
  tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(10)
])

# Train the digit classification model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])
model.fit(
  train_images,
  train_labels,
  epochs=1,
  validation_data=(test_images, test_labels)
)

"""## 转换为 TensorFlow Lite 模型

现在，您可以使用 [`TFLiteConverter`](https://tensorflow.google.cn/lite/convert/python_api) API 将训练的模型转换为 TensorFlow Lite 格式，并应用不同程度的量化。

请注意，某些版本的量化会将部分数据保留为浮点格式。因此，以下各个部分将以量化程度不断增加的顺序展示每个选项，直到获得完全由 int8 或 uint8 数据组成的模型。（请注意，我们在每个部分中重复了一些代码，使您能够看到每个选项的全部量化步骤。）

首先，下面是一个没有量化的转换后模型：
"""

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

"""它现在是一个 TensorFlow Lite 模型，但所有参数数据仍使用 32 位浮点值。

### 使用动态范围量化进行转换

现在，我们启用默认的 `optimizations` 标记来量化所有固定参数（例如权重）：
"""

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model_quant = converter.convert()

"""现在，进行了权重量化的模型要略小一些，但其他变量数据仍为浮点格式。

### 使用浮点回退量化进行转换

要量化可变数据（例如模型输入/输出和层之间的中间体），您需要提供 [`RepresentativeDataset`](https://tensorflow.google.cn/api_docs/python/tf/lite/RepresentativeDataset)。这是一个生成器函数，它提供一组足够大的输入数据来代表典型值。转换器可以通过该函数估算所有可变数据的动态范围。（相比训练或评估数据集，此数据集不必唯一。）为了支持多个输入，每个代表性数据点都是一个列表，并且列表中的元素会根据其索引被馈送到模型。
"""

def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
    # Model has only one input so each data point has one element.
    yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

tflite_model_quant = converter.convert()

"""现在，所有权重和可变数据都已量化，并且与原始 TensorFlow Lite 模型相比，该模型要小得多。

但是，为了与传统上使用浮点模型输入和输出张量的应用保持兼容，TensorFlow Lite 转换器将模型的输入和输出张量保留为浮点：
"""

interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)

"""这通常对兼容性有利，但它无法兼容执行全整数运算的设备（如 Edge TPU）。

此外，如果 TensorFlow Lite 不包括某个运算的量化实现，则上述过程可能会将该运算保留为浮点格式。您仍能通过此策略完成转换，并得到一个更小、更高效的模型，但它还是不兼容仅支持整数的硬件。（此 MNIST 模型中的所有算子都有量化的实现。）

因此，为了确保端到端全整数模型，您还需要几个参数…

### 使用仅整数量化进行转换

为了量化输入和输出张量，并让转换器在遇到无法量化的运算时引发错误，使用一些附加参数再次转换模型：
"""

def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
    yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model_quant = converter.convert()

"""内部量化与上文相同，但您可以看到输入和输出张量现在是整数格式：

"""

interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)

"""现在，您有了一个整数量化模型，该模型使用整数数据作为模型的输入和输出张量，因此它兼容仅支持整数的硬件（如 [Edge TPU](https://coral.ai)）。

### 将模型另存为文件

您需要 `.tflite` 文件才能在其他设备上部署模型。因此，我们将转换的模型保存为文件，然后在下面运行推断时加载它们。
"""

import pathlib

tflite_models_dir = pathlib.Path("./mnist_tflite_models/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

# Save the unquantized/float model:
tflite_model_file = tflite_models_dir/"mnist_model.tflite"
tflite_model_file.write_bytes(tflite_model)

# Save the quantized model:
tflite_model_quant_file = tflite_models_dir/"mnist_model_quant.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant)


"""## 运行 TensorFlow Lite 模型

现在，我们使用 TensorFlow Lite [`Interpreter`](https://tensorflow.google.cn/api_docs/python/tf/lite/Interpreter) 运行推断来比较模型的准确率。

首先，我们需要一个函数，该函数使用给定的模型和图像运行推断，然后返回预测值：
"""

# Helper function to run inference on a TFLite model
def run_tflite_model(tflite_file, test_image_indices):
  global test_images

  # Initialize the interpreter
  interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  predictions = np.zeros((len(test_image_indices),), dtype=int)
  for i, test_image_index in enumerate(test_image_indices):
    test_image = test_images[test_image_index]
    test_label = test_labels[test_image_index]

    # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8:
      input_scale, input_zero_point = input_details["quantization"]
      test_image = test_image / input_scale + input_zero_point

    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], test_image)


    if "mnist_model_quant.tflite" in str(tflite_file):
      np_array8 = interpreter.get_tensor(8)
      # print("david", np_array8)
      interpreter.set_tensor(8, np_array8)
      interpreter.reset_all_variables()

    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    # Check if the output type is quantized, then rescale output data to float
    if output_details['dtype'] == np.uint8:
      output_scale, output_zero_point = output_details["quantization"]
      test_image = test_image.astype(np.float32)
      test_image = test_image / input_scale + input_zero_point

    predictions[i] = output.argmax()

  return predictions

"""### 在单个图像上测试模型

现在，我们来比较一下浮点模型和量化模型的性能：

- `tflite_model_file` 是使用浮点数据的原始 TensorFlow Lite 模型。
- `tflite_model_quant_file` 是我们使用全整数量化转换的上一个模型（它使用 uint8 数据作为输入和输出）。

我们来创建另一个函数打印预测值：
"""

import matplotlib.pylab as plt

# Change this to test a different image
test_image_index = 1

## Helper function to test the models on one image
def test_model(tflite_file, test_image_index, model_type):
  global test_labels

  predictions = run_tflite_model(tflite_file, [test_image_index])

  plt.imshow(test_images[test_image_index])
  template = model_type + " Model \n True:{true}, Predicted:{predict}"
  _ = plt.title(template.format(true= str(test_labels[test_image_index]), predict=str(predictions[0])))
  plt.grid(False)

# test_model(tflite_model_file, test_image_index, model_type="Float")

# test_model(tflite_model_quant_file, test_image_index, model_type="Quantized")

# Helper function to evaluate a TFLite model on all images
def evaluate_model(tflite_file, model_type):
  global test_images
  global test_labels

  test_image_indices = range(test_images.shape[0])
  predictions = run_tflite_model(tflite_file, test_image_indices)

  accuracy = (np.sum(test_labels== predictions) * 100) / len(test_images)

  print('%s model accuracy is %.4f%% (Number of test samples=%d)' % (
      model_type, accuracy, len(test_images)))


# evaluate_model(tflite_model_file, model_type="Float")
print("BAD!!!!!!!!!!!!")
evaluate_model(tflite_model_quant_file, model_type="Quantized")
print("OVER!!!!!!!!!!!")
