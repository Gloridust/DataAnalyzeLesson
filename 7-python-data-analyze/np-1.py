import numpy as np
import time

def take_sub_deter(local_deter, row_to_delete, column_to_delete):
    """
    从给定矩阵中删除指定的行和列，生成子矩阵
    
    参数:
    local_deter: 原始矩阵
    row_to_delete: 要删除的行索引
    column_to_delete: 要删除的列索引
    
    返回:
    sub_deter: 生成的子矩阵
    """
    sub_deter = []
    for i in range(len(local_deter)):
        if i != row_to_delete:  # 跳过要删除的行
            new_row = []
            for j in range(len(local_deter[i])):
                if j != column_to_delete:  # 跳过要删除的列
                    new_row.append(local_deter[i][j])
            sub_deter.append(new_row)
    return sub_deter

def calc_deter(the_deter):
    """
    递归计算矩阵的行列式
    
    参数:
    the_deter: 要计算行列式的矩阵
    
    返回:
    det_value: 计算得到的行列式值
    """
    n = len(the_deter)
    
    # 基本情况：1x1 矩阵
    if n == 1:
        return the_deter[0][0]
    
    # 基本情况：2x2 矩阵
    if n == 2:
        return the_deter[0][0] * the_deter[1][1] - the_deter[0][1] * the_deter[1][0]
    
    # 递归情况：nxn 矩阵 (n > 2)
    det_value = 0
    for j in range(n):
        # 生成子矩阵，删除第一行和第j列
        sub_det = take_sub_deter(the_deter, 0, j)
        # 计算代数余子式
        cofactor = the_deter[0][j] * ((-1) ** j)
        # 递归计算子矩阵的行列式，并累加到结果中
        det_value += cofactor * calc_deter(sub_det)
    
    return det_value

matrix_12x12 = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3],
    [4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6],
    [7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3],
    [4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6],
    [7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3],
    [4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6],
    [7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3],
    [4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6],
    [7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9]
]

# 自定义函数计算
start_time = time.time()
result_custom = calc_deter(matrix_12x12)
end_time = time.time()
custom_time = end_time - start_time
print(f"自定义函数计算 12 阶行列式的时间: {custom_time} 秒")

# numpy 计算
start_time = time.time()
result_numpy = np.linalg.det(np.array(matrix_12x12))
end_time = time.time()
numpy_time = end_time - start_time
print(f"numpy 计算 12 阶行列式的时间: {numpy_time} 秒")
