import numpy as np  # 导入NumPy库并简写为np

# 1. 创建NumPy数组（ndarray）
# 与Python列表相比，NumPy数组支持批量操作
arr1 = np.array([1, 2, 3, 4, 5])                                            # 一维数组
arr2 = np.array([6, 7, 8, 9, 10])

# 2. 查看数组基本信息
print("数组1:", arr1)
print("数组1的形状:", arr1.shape)                                            # 输出 (5,) 表示一维数组，有5个元素
print("数组1的数据类型:", arr1.dtype)                                         # 输出 int64（默认整数类型）

# 3. 数组批量运算（无需循环，直接对每个元素操作）
print("\n数组相加:", arr1 + arr2)                                           # 对应元素相加 [7 9 11 13 15]
print("数组相乘:", arr1 * arr2)                                             # 对应元素相乘 [6 14 24 36 50]
print("数组乘以2:", arr1 * 2)                                               # 每个元素都乘以2 [2 4 6 8 10]

# 4. 统计计算
print("\n数组1的平均值:", arr1.mean())                                       # 计算平均值 3.0
print("数组1的总和:", arr1.sum())                                            # 计算总和 15
print("数组1的最大值:", arr1.max())                                          # 找出最大值 5
print("数组1的最小值:", arr1.min())                                          # 找出最小值 1

# 5. 二维数组示例（类似矩阵）
arr3 = np.array([[1, 2], [3, 4], [5, 6]])                                  # 3行2列的二维数组
print("\n二维数组:\n", arr3)
print("二维数组的形状:", arr3.shape)                                         # 输出 (3, 2) 表示3行2列
print("二维数组的每行平均值:", arr3.mean(axis=1))                              # 按行计算平均值 [1.5 3.5 5.5]
