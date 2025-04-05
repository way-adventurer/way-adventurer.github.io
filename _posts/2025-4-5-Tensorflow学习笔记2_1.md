---
title: TensorFlow学习笔记(2.1)：BasicModels章linear_regression篇
author: way-adventurer
date: 2025-04-05 23:38:00 +0800
categories: [教程]
tags: [TensorFlow, Python, 线性回归]
render_with_liquid: false
image:
  path: /assets/img/posts/20250405/cover1.png
  alt: TensorFlow基础学习
---

> 本文是 TensorFlow 学习笔记系列的实操第二篇，主要介绍 TensorFlow 的线性回归。通过实际代码示例，帮助初学者快速入门 TensorFlow 2.x。

## 1. 线性回归简介
线性回归是机器学习中最基础的算法之一，用于预测连续型数值。本教程将使用 TensorFlow 2.x 实现一个简单的线性回归模型。

## 2. 环境准备
```python
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
rng = np.random
```

## 3. 模型参数设置
```python
# 模型超参数
learning_rate = 0.01    # 学习率
training_steps = 1000   # 训练步数
display_step = 50      # 显示间隔
```

## 4. 数据准备
```python
# 训练数据
X = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167])
Y = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221])
```

## 5. 模型构建
### 5.1 初始化参数
```python
# 初始化权重和偏置
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# 线性回归模型
def linear_regression(x):
    return W * x + b
```

### 5.2 损失函数
```python
# 均方误差
def mean_square(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))
```

#### 函数详解
1. **tf.square**
   - 功能：计算张量的平方
   - 参数：接受一个张量输入
   - 返回：每个元素平方后的张量
   - 示例：
     ```python
     x = tf.constant([2.0, 3.0])
     square_x = tf.square(x)  # 结果: [4.0, 9.0]
     ```

2. **tf.reduce_mean**
   - 功能：计算张量所有元素的平均值
   - 参数：
     - input_tensor：输入张量
     - axis：计算平均值的维度（可选）
     - keepdims：是否保持维度（默认False）
   - 示例：
     ```python
     x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
     mean_all = tf.reduce_mean(x)  # 所有元素平均: 2.5
     mean_axis0 = tf.reduce_mean(x, axis=0)  # 按列平均: [2.0, 3.0]
     ```

## 6. 模型训练
```python
# 随机梯度下降优化器
optimizer = tf.optimizers.SGD(learning_rate)
```

### 优化器详解
**tf.optimizers.SGD**
- 功能：实现随机梯度下降算法
- 主要参数：
  - learning_rate：学习率，控制参数更新步长
  - momentum：动量参数（可选）
  - nesterov：是否使用Nesterov动量（可选）
- 示例：
  ```python
  # 基本SGD优化器
  basic_sgd = tf.optimizers.SGD(learning_rate=0.01)
  
  # 带动量的SGD优化器
  momentum_sgd = tf.optimizers.SGD(
      learning_rate=0.01,
      momentum=0.9
  )
  ```

## 7. 优化过程
```python
def run_optimization():
    with tf.GradientTape() as g:
        # 前向传播
        pred = linear_regression(X)
        # 计算预测值 pred 和真实值 Y 之间的均方误差
        loss = mean_square(pred, Y)  
    # 计算损失 loss 对模型参数 [W, b] 的梯度
    gradients = g.gradient(loss, [W, b])
    # 使用优化器（optimizer）根据计算出的梯度更新模型参数 [W, b]，之前定义使用的优化算法为随机梯度下降（SGD）
    optimizer.apply_gradients(zip(gradients, [W, b]))
```

### GradientTape详解
**tf.GradientTape**
- 功能：自动记录梯度计算所需的操作
- 主要特点：
  - 上下文管理器（with语句）
  - 自动微分机制
  - 可以计算一阶和高阶导数
- 示例：
  ```python
    x = tf.Variable(3.0)  # 创建一个可训练变量，初始值为3.0
    with tf.GradientTape() as tape:  # 创建梯度记录器
        y = x * x  # 定义计算图：y = x^2
    dy_dx = tape.gradient(y, x)  # 计算导数：dy/dx
  ```
- 完整运行示例：
  ```python
    import tensorflow as tf

    x = tf.Variable(3.0)
    with tf.GradientTape() as tape:
        y = x * x
    dy_dx = tape.gradient(y, x)
    print(f"x = {x.numpy()}")
    print(f"y = {y.numpy()}")
    print(f"dy/dx = {dy_dx.numpy()}")
  ```
- 输出结果：
  ```python
    x = 3.0
    y = 9.0
    dy/dx = 6.0
  ```
- 注意事项：
    - GradientTape 默认只能使用一次
    - 如需多次使用，需要设置 persistent=True
    - 计算完成后应及时释放资源
    - 只有 tf.Variable 类型的变量才能计算梯度

## 8. 关键概念解释
1. **梯度带(GradientTape)**
   - 自动微分机制
   - 记录操作用于反向传播

2. **优化器(Optimizer)**
   - 实现参数更新
   - 控制学习过程

3. **损失函数**
   - 评估模型性能
   - 指导优化方向

## 9. 实践技巧
1. **学习率选择**
   - 太大可能导致不收敛
   - 太小会导致收敛慢

2. **训练步数设置**
   - 根据损失变化调整
   - 避免过拟合

3. **数据预处理**
   - 特征缩放
   - 异常值处理

## 10. 模型训练过程
```python
# Run training for the given number of steps.
for step in range(1, training_steps + 1):
    # Run the optimization to update W and b values.
    run_optimization()
    
    if step % display_step == 0:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)
        print("step: %i, loss: %f, W: %f, b: %f" % (step, loss, W.numpy(), b.numpy()))
```
- **`training_steps`**：指定训练的总步数。（此案例为1000）
- **`step`**：表示当前的训练步数，从 1 开始到 `training_steps` 结束。

### 10.1 运行优化函数
在每次训练步中，调用 `run_optimization()` 函数执行一次优化步骤：
```python
    # Run the optimization to update W and b values.
    run_optimization()
```

#### 优化过程详解
- **`GradientTape`**：记录计算图中的操作，用于自动计算梯度。
- **梯度更新**：根据计算出的梯度，使用优化器（如 SGD）更新模型参数 `W` 和 `b`。
- 每次调用 `run_optimization()` 都会根据当前的损失函数值调整权重和偏置。

### 10.2 输出训练进度
为了监控训练过程，每隔 `display_step` 步数输出当前的训练状态：
```python
    if step % display_step == 0:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)
        print("step: %i, loss: %f, W: %f, b: %f" % (step, loss, W.numpy(), b.numpy()))
```

#### 具体步骤
1. **预测值计算**：
   ```python
   pred = linear_regression(X)
   ```
   - 调用 `linear_regression` 函数计算输入数据 `X` 的预测值。
   - 预测公式为：pred = W * X + b。

2. **损失值计算**：
   ```python
   loss = mean_square(pred, Y)
   ```
   - 调用 `mean_square` 函数计算预测值 `pred` 和真实值 `Y` 之间的均方误差（MSE）。

3. **打印训练信息**：
   ```python
   print("step: %i, loss: %f, W: %f, b: %f" % (step, loss, W.numpy(), b.numpy()))
   ```
   - 打印当前步数、损失值以及模型参数 `W` 和 `b` 的值。
   - 使用 `W.numpy()` 和 `b.numpy()` 将 TensorFlow 变量转换为 NumPy 数组以便打印。

### 10.3 输出训练进度结果
```python
step: 50, loss: 0.421195, W: 0.458914, b: -0.670712
step: 100, loss: 0.363510, W: 0.435194, b: -0.502547
step: 150, loss: 0.318272, W: 0.414188, b: -0.353626
step: 200, loss: 0.282795, W: 0.395586, b: -0.221747
step: 250, loss: 0.254974, W: 0.379113, b: -0.104960
step: 300, loss: 0.233155, W: 0.364525, b: -0.001537
step: 350, loss: 0.216045, W: 0.351606, b: 0.090051
step: 400, loss: 0.202626, W: 0.340166, b: 0.171157
step: 450, loss: 0.192103, W: 0.330035, b: 0.242982
step: 500, loss: 0.183851, W: 0.321063, b: 0.306588
step: 550, loss: 0.177379, W: 0.313118, b: 0.362915
step: 600, loss: 0.172304, W: 0.306082, b: 0.412796
step: 650, loss: 0.168323, W: 0.299851, b: 0.456969
step: 700, loss: 0.165202, W: 0.294334, b: 0.496087
step: 750, loss: 0.162754, W: 0.289447, b: 0.530728
step: 800, loss: 0.160835, W: 0.285120, b: 0.561405
step: 850, loss: 0.159329, W: 0.281288, b: 0.588572
step: 900, loss: 0.158148, W: 0.277895, b: 0.612630
step: 950, loss: 0.157223, W: 0.274890, b: 0.633934
step: 1000, loss: 0.156497, W: 0.272229, b: 0.652801
```

## 11. 可视化结果
```python
import matplotlib.pyplot as plt

plt.plot(X, Y, 'ro', label='原始数据')
plt.plot(X, np.array(W * X + b), label='拟合线')
plt.legend()
plt.show()
```

### matplotlib绘图详解
1. **plt.plot**
   - 功能：绘制二维图形
   - 常用参数：
     - x, y：数据点坐标
     - 'ro'：格式字符串（'r'表示红色，'o'表示圆点）
     - label：图例标签
   - 示例：
     ```python
     # 绘制散点图
     plt.plot([1,2,3], [1,2,3], 'ro')
     
     # 绘制线图
     plt.plot([1,2,3], [1,2,3], '-b', label='线条')
     ```

2. **plt.legend**
   - 功能：显示图例
   - 常用参数：
     - loc：图例位置（'best', 'upper right'等）
     - shadow：是否显示阴影
   - 示例：
     ```python
     plt.legend(loc='best', shadow=True)
     ```

## 12. 推荐学习顺序（BasicModels章）
1. 线性回归 (linear_regression.ipynb)
- 最基础和简单的模型
- 介绍核心概念:
  - 张量(Tensor)
  - 变量(Variable)
  - 优化器(Optimizer)
  - 损失函数(Loss Function)
- 代码结构清晰,容易理解
- 包含可视化结果部分
2. 逻辑回归 (logistic_regression.ipynb)
- 基于线性回归,增加了分类任务
- 使用经典的 MNIST 数据集
- 引入新概念:
  - 交叉熵损失
  - Softmax 函数
  - 分类准确率评估
- 难度适中,是分类问题的基础
3. 梯度提升决策树 (gradient_boosted_trees.md)
- 介绍更复杂的集成学习模型
- 同时涵盖:
  - 分类任务
  - 回归任务
- 使用 TensorFlow 高级 API
- 实际应用案例:波士顿房价预测
4. Word2Vec (word2vec.ipynb)
- NLP 领域的经典模型
- 较复杂的概念:
  - 词嵌入(Word Embedding)
  - Skip-gram 模型
  - 负采样(Negative Sampling)
- 建议掌握基础后再学习

## 参考资源
- TensorFlow 官方文档：https://www.tensorflow.org/
- GitHub 项目：https://github.com/aymericdamien/TensorFlow-Examples/
