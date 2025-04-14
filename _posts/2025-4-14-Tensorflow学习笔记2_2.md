---
title: TensorFlow学习笔记(2.2)：BasicModels章logistic_regression篇
author: way-adventurer
date: 2025-04-14 22:17:00 +0800
categories: [教程]
tags: [TensorFlow, Python, 逻辑回归]
render_with_liquid: false
image:
  path: /assets/img/posts/20250414/cover.png
  alt: TensorFlow基础学习
---

> 本文是 TensorFlow 学习笔记系列的实操第二篇，将详细讲解如何使用 TensorFlow 2.x 实现逻辑回归模型来识别手写数字。我们将使用 MNIST 数据集，这是机器学习领域最基础和著名的数据集之一。

## 2. MNIST数据集介绍

MNIST数据集包含：
- 训练集：60,000张图片
- 测试集：10,000张图片
- 图片尺寸：28x28像素
- 数据特征：灰度图像，像素值范围0-255
- 分类目标：10个数字（0-9）

## 3. 环境准备

导入库函数：
```python
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
```

## 4. 代码详解

### 4.1 数据预处理
```python
# 数据集参数
num_classes = 10    # 分类数量（0-9）
num_features = 784  # 特征数量（28*28=784）

# 训练参数
learning_rate = 0.01      # 学习率
training_steps = 1000     # 训练步数
batch_size = 256          # 批次大小
display_step = 50         # 每隔多少步显示一次结果
```

学习率（learning_rate）是机器学习和深度学习中优化算法的一个超参数，用于控制模型参数更新的步长。在梯度下降法中，学习率决定了沿着梯度方向每次迭代时参数更新的步伐大小。`learning_rate = 0.01`表示每次参数更新时，调整的幅度为当前梯度值的百分之一点。

学习率的影响：
1. 过大的学习率：可能导致损失函数值波动较大，无法稳定收敛，甚至可能发散。
2. 过小的学习率：虽然可以更精确地接近最优解，但会导致训练过程非常缓慢，增加训练时间。

```python
# 准备MNIST数据集
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 转换为float32类型
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# 展平为二维图像
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
# 归一化像素值
x_train, x_test = x_train / 255., x_test / 255.
```

数据预处理步骤：
1. 加载MNIST数据集，使用`x_train`、`y_train`作为训练集，`x_test`、`y_test`作为测试集
2. 将图像数据转换为float32类型，确保后续计算时具有更高的精度。
3. 将28x28的图像从原始的二维图像格式转换为适合输入到逻辑回归模型的一维特征向量，`-1` 表示自动计算该维度的大小（即样本数保持不变）；`num_features` 是展平后的特征数量，等于图像的高度乘以宽度
4. 将像素值归一化到[0,1]区间

```python
# Use tf.data API to shuffle and batch data.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
```

使用 TensorFlow 的 `tf.data` API 来处理训练数据集，使其更适合模型训练。
1. 使用 `tf.data.Dataset.from_tensor_slices` 方法将训练数据 `x_train` 和标签 `y_train` 组合成一个数据集。`x_train` 包含图像数据，`y_train` 包含对应的标签。
2. 无限重复数据集以支持多次遍历。打乱数据集以确保训练过程中的随机性。将数据集分成小批量，以便高效训练。预取数据以减少 I/O 延迟，提高训练速度。

- `repeat()` 表示重复数据集，这意味着在训练过程中，数据集会不断循环使用，直到达到指定的训练步数 `training_steps`。
- `shuffle(5000)` 表示打乱数据集，参数 `5000` 表示打乱时使用的缓冲区大小。较大的缓冲区可以更好地打乱数据，但会占用更多的内存。
- `batch(batch_size)` 表示将数据集批量化，`batch_size` 是每个批量的大小，这里设置为 `256`。批量处理可以提高训练效率，因为 GPU 和 CPU 可以并行处理多个样本。
- `prefetch(1)` 表示预取数据预取可以在训练模型的同时准备下一批数据，从而减少 I/O 操作的延迟，提高训练速度。
参数 `1` 表示预取一个批量的数据。

### 4.2 逻辑回归模型实现

#### 模型参数
```python
W = tf.Variable(tf.ones([num_features, num_classes]), name="weight")
b = tf.Variable(tf.zeros([num_classes]), name="bias")
```
**权重 `W` 的定义**：
- `tf.Variable`：创建一个 TensorFlow 变量，用于存储模型的可训练参数。
- `tf.ones([num_features, num_classes])`：初始化权重矩阵 `W`，其形状为 `[num_features, num_classes]`，即 `[784, 10]`。这里使用全 1 的张量进行初始化。
  - `num_features = 784`：表示每个 MNIST 图像被展平后的特征数（28x28=784）。
  - `num_classes = 10`：表示分类任务的类别数（MNIST 数据集有 10 个数字类别：0-9）。
- `name="weight"`：为变量指定名称，便于调试和可视化。

**偏置 `b` 的定义**：
- `tf.Variable`：同样创建一个 TensorFlow 变量。
- `tf.zeros([num_classes])`：初始化偏置向量 `b`，其形状为 `[num_classes]`，即 `[10]`。这里使用全 0 的张量进行初始化。
- `name="bias"`：为变量指定名称。

#### 核心函数解析：

1. **逻辑回归函数**
```python
def logistic_regression(x):
    return tf.nn.softmax(tf.matmul(x, W) + b)
```

`x`: 输入特征矩阵，形状为 `[batch_size, num_features]`。

**矩阵乘法 (`tf.matmul(x, W)`)**:
- `tf.matmul(x, W)` 计算输入特征 `x` 和权重矩阵 `W` 的矩阵乘法。
- 结果的形状为 `[batch_size, num_classes]`。每个输入样本会被映射到 10 个类别的得分（logits）。

**偏置加法 (`+ b`)**:
- `+ b` 将偏置向量 `b` 加到每个样本的 logits 上。
- `b` 的形状为 `[num_classes]`，偏置会被广播到 `[batch_size, num_classes]` 的形状。

**Softmax 函数 (`tf.nn.softmax(...)`)**:
- `tf.nn.softmax(...)` 将 logits 转换为概率分布。
- Softmax 函数的输出是一个形状为 `[batch_size, num_classes]` 的矩阵，其中每一行的元素之和为 1，表示每个样本属于各个类别的概率。

返回值是一个形状为 `[batch_size, num_classes]` 的矩阵，其中每个元素表示对应样本属于某个类别的概率。

2. **交叉熵损失函数**
```python
def cross_entropy(y_pred, y_true):
    y_true = tf.one_hot(y_true, depth=num_classes)
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred),1))
```

`y_pred`: 模型预测的概率分布，形状为 `[batch_size, num_classes]`。每个元素表示对应样本属于某个类别的概率。

`y_true`: 真实标签，形状为 `[batch_size]`。每个元素表示对应样本的真实类别（整数）。

**将真实标签转换为 one-hot 编码**
- `tf.one_hot(y_true, depth=num_classes)` 将真实标签 `y_true` 转换为 one-hot 编码。
- `y_true` 的形状从 `[batch_size]` 变为 `[batch_size, num_classes]`。
- 例如，如果 `y_true` 是 `[3, 1, 2]`，`num_classes` 是 `10`，那么 `y_true` 将被转换为：
  ```python
  [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # index=3处值为1
  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],   # index=1处值为1
  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]   # index=2处值为1
  ```

**裁剪预测概率**
- `tf.clip_by_value(y_pred, 1e-9, 1.)` 将 `y_pred` 中的值裁剪到 `[1e-9, 1]` 范围内。
- 这一步是为了防止在计算 `log(y_pred)` 时出现 `log(0)` 的情况，避免数值不稳定或 NaN 值。如果 `y_pred` 中有值接近 0，裁剪后这些值将变为 `1e-9`。

**计算交叉熵损失**
- `tf.math.log(y_pred)` 计算 `y_pred` 中每个元素的自然对数。
- `y_true * tf.math.log(y_pred)` 计算每个样本的真实类别对应的预测概率的对数。
- `tf.reduce_sum(..., 1)` 按照类别维度（即第二个维度）求和，得到每个样本的交叉熵损失。
- `-tf.reduce_sum(..., 1)` 取负值，得到每个样本的交叉熵损失。
- `tf.reduce_mean(...)` 计算所有样本的交叉熵损失的平均值。

返回值是一个标量，表示整个批次的平均交叉熵损失。

3. **模型准确率函数**
```python
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

`y_pred`: 模型预测的概率分布，形状为 `[batch_size, num_classes]`。每个元素表示对应样本属于某个类别的概率。

`y_true`: 真实标签，形状为 `[batch_size]`。每个元素表示对应样本的真实类别（整数）。

**预测类别**
- `tf.argmax(y_pred, 1)` 返回 `y_pred` 中每个样本预测概率最高的类别索引（即预测类别）。
- 例如，如果 `y_pred` 是 `[[0.1, 0.2, 0.7], [0.6, 0.2, 0.2]]`，`tf.argmax(y_pred, 1)` 将返回 `[2, 0]`，表示第一个样本预测为类别 2，第二个样本预测为类别 0。

**比较预测类别与真实类别**
- `tf.cast(y_true, tf.int64)` 将真实标签 `y_true` 转换为 `int64` 类型，以确保与 `tf.argmax(y_pred, 1)` 的输出类型一致。
- `tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))` 比较预测类别与真实类别，返回一个布尔张量 `correct_prediction`，其中每个元素表示对应样本的预测是否正确。
- 例如，如果`tf.argmax(y_pred, 1)` 返回 `[2, 0]`， `y_true` 是 `[2, 1]`，`correct_prediction` 将返回 `[True, False]`，表示第一个样本预测正确，第二个样本预测错误。

**计算准确率**
- `tf.cast(correct_prediction, tf.float32)` 将布尔张量 `correct_prediction` 转换为 `float32` 类型，其中 `True` 转换为 `1.0`，`False` 转换为 `0.0`。
- `tf.reduce_mean(tf.cast(correct_prediction, tf.float32))` 计算转换后的张量的平均值，即准确率。
- 例如，如果 `correct_prediction` 是 `[True, False]`，转换后的张量将是 `[1.0, 0.0]`，平均值为 `0.5`，表示准确率为 50%。

返回值是一个标量，表示模型在给定批次上的准确率。

## 5. 训练过程详解

### 5.1 优化器
```python
optimizer = tf.optimizers.SGD(learning_rate)
```
使用随机梯度下降(SGD)优化器，学习率为0.01

### 5.2 训练过程详解

```python
def run_optimization(x, y):
    with tf.GradientTape() as g:
        pred = logistic_regression(x)
        loss = cross_entropy(pred, y)
    gradients = g.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))
```

**GradientTape 的使用**
- `tf.GradientTape()` 是 TensorFlow 的自动微分机制
- 作用：记录计算过程以便自动计算梯度
- 工作流程：
  1. 在 tape 的上下文中执行前向传播
  2. 计算损失值
  3. 使用 tape 计算梯度
  4. 应用梯度更新参数

**优化步骤详解**：

前向传播：`pred = logistic_regression(x)`，将输入数据传入模型得到预测值。

计算损失：`loss = cross_entropy(pred, y)`，使用交叉熵计算预测值与真实值的差异。

计算梯度：`gradients = g.gradient(loss, [W, b])`，自动计算损失对权重和偏置的梯度。

更新参数：`optimizer.apply_gradients(zip(gradients, [W, b]))`，使用优化器根据梯度更新模型参数。

```python
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    run_optimization(batch_x, batch_y)
    if step % display_step == 0:
        pred = logistic_regression(batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
```

**训练循环组成部分**：

- `train_data.take(training_steps)`：获取指定步数的训练数据。
- `enumerate(..., 1)`：生成从1开始的步数计数。
- 定期监控：
  - 每 `display_step` 步输出一次训练状态
  - 输出内容包括：当前步数、损失值、准确率

## 6. 模型评估与可视化

### 6.1 模型测试
```python
pred = logistic_regression(x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))
```
- 在完整测试集上评估模型性能
- 使用准确率作为评估指标
- 无需分批处理，直接使用全部测试数据

### 6.2 预测结果可视化
```python
import matplotlib.pyplot as plt

n_images = 5
test_images = x_test[:n_images]
predictions = logistic_regression(test_images)

for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()
    print("Model prediction: %i" % np.argmax(predictions.numpy()[i]))
```

**可视化步骤解析**：
1. 选择要显示的图片数量：`n_images = 5`
2. 获取测试图片：`test_images = x_test[:n_images]`
3. 进行预测：`predictions = logistic_regression(test_images)`
4. 显示图片和预测结果：
   - `plt.imshow()` 显示原始图片
   - `np.reshape()` 将展平的图片重新变为28x28
   - `np.argmax()` 获取预测的数字类别

## 7.TensorFlow核心函数详解

#### tf.matmul 函数
```python
tf.matmul(x, W)  # 矩阵乘法运算
```
- 功能：实现两个张量的矩阵乘法
- 参数说明：
  - x：输入数据，shape为[batch_size, 784]
  - W：权重矩阵，shape为[784, 10]
- 返回值：shape为[batch_size, 10]的矩阵
- 使用场景：在神经网络中进行线性变换

#### tf.nn.softmax 函数
```python
tf.nn.softmax(logits)  # 将logits转换为概率分布
```
- 功能：将神经网络输出转换为概率分布
- 计算公式：softmax(x_i) = exp(x_i) / Σ exp(x_j)
- 特点：
  - 输出值范围在[0,1]之间
  - 所有输出值的和为1
  - 常用于多分类问题

#### tf.one_hot 函数
```python
tf.one_hot(indices, depth=num_classes)
```
- 功能：将标签转换为独热编码
- 参数说明：
  - indices：原始标签，如[3,1,2]
  - depth：编码维度，这里是10（数字0-9）
- 示例：
  ```python
  # 输入：2
  # 输出：[0,0,1,0,0,0,0,0,0,0]
  ```

#### tf.clip_by_value 函数
```python
tf.clip_by_value(y_pred, 1e-9, 1.)
```
- 功能：将张量的值裁剪到指定范围
- 参数说明：
  - tensor：需要裁剪的张量
  - min：最小值（这里是1e-9）
  - max：最大值（这里是1.0）
- 使用原因：防止计算log(0)导致的数值问题

#### tf.reduce_mean 和 tf.reduce_sum 函数
```python
tf.reduce_mean(input)  # 计算平均值
tf.reduce_sum(input, axis=1)  # 按指定轴求和
```
- reduce_mean：
  - 功能：计算张量所有元素的平均值
  - 常用于计算损失值
- reduce_sum：
  - 功能：计算张量在指定维度上的和
  - axis参数指定求和的维度

#### tf.data.Dataset
```python
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
```
- 功能：创建数据集对象
- 主要方法：
  - repeat()：重复数据集
  - shuffle(buffer_size)：随机打乱数据
  - batch(batch_size)：设置批次大小
  - prefetch(1)：预加载下一个批次

#### tf.Variable
```python
W = tf.Variable(tf.ones([num_features, num_classes]))
```
- 功能：创建可训练的变量
- 特点：
  - 可以被优化器更新
  - 在训练过程中保持状态
  - 可以保存和恢复

## 8. 实用技巧与注意事项

### 8.1 性能优化技巧
1. **数据预处理优化**：
   - 使用 `tf.data.Dataset` 进行数据管理
   - 合理设置 `prefetch` 和 `shuffle` 参数
   - 适当的批次大小选择

2. **训练过程优化**：
   - 定期监控损失和准确率
   - 适时调整学习率
   - 注意梯度爆炸和消失问题

3. **内存管理**：
   - 避免一次性加载过多数据
   - 及时释放不需要的变量
   - 使用 `tf.function` 优化计算图

### 8.2 调试建议
1. **常见问题排查**：
   - 损失值未下降：检查学习率和梯度计算
   - 准确率波动：检查数据预处理和批次大小
   - 内存溢出：调整批次大小和数据加载方式

2. **模型监控**：
   - 使用 TensorBoard 可视化训练过程
   - 保存关键节点的中间结果
   - 记录重要参数的变化趋势

## 9. 扩展与改进方向

1. **模型改进**：
   - 添加正则化层
   - 使用更复杂的神经网络结构
   - 实现早停机制

2. **功能扩展**：
   - 添加模型保存和加载功能
   - 实现交叉验证
   - 添加更多评估指标

3. **实践应用**：
   - 应用到其他数据集
   - 处理实际业务场景
   - 部署模型服务
