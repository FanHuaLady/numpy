import numpy as np

# ------------------------------------------------------# 本次实验用到的数据
sample = np.array([[0,0],[0,1],[1,0],[1,1]])            # 样本
answer = np.array([[0], [0], [0], [1]])                 # 样本对应的答案

# ------------------------------------------------------# 本次实验的神经网络参数
np.random.seed(1)                                       # 固定随机种子
weight = np.random.randn(2, 1)                          # 权重矩阵
bias = np.random.randn(1)                               # 偏置

# ------------------------------------------------------# 将输出映射到0-1之间
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

learning_rate = 0.1                                     # 学习率（控制参数更新幅度）
epochs = 10000                                          # 训练轮次【循环次数】

for i in range(epochs):
    z = np.dot(sample, weight) + bias                   # 线性变换：z = X·W + b
    temp_pred = sigmoid(z)                              # 获得数字，这个数字可能近0，也可能近1

    # 猜的越准loss越小，即损失越小
    loss = -np.mean(answer * np.log(temp_pred) + (1 - answer) * np.log(1 - temp_pred))

    residual = temp_pred - answer                       # 模型预测结果与真实标签之间的误差方向和大小
    repair_weight = np.dot(sample.T, residual) / 4      # 对权重的修补
    repair_bias = np.mean(residual)                     # 对偏移的修补

    weight -= learning_rate * repair_weight
    bias -= learning_rate * repair_bias

    if i % 1000 == 0:
        print(f"第{i}轮，损失：{loss:.4f}")

print("\n训练完成后，预测结果：")
z = np.dot(sample, weight) + bias
y_pred = sigmoid(z)

for i in range(4):
    print(f"输入：{sample[i]}，预测概率：{y_pred[i][0]:.4f}，预测标签：{1 if y_pred[i][0] > 0.5 else 0}")