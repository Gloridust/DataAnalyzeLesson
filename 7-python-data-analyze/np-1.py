import numpy as np
import time

def take_sub_deter(local_deter, row_to_delete, column_to_delete):
    sub_deter = []
    for i in range(len(local_deter)):
        if i != row_to_delete:
            new_row = []
            for j in range(len(local_deter[i])):
                if j != column_to_delete:
                    new_row.append(local_deter[i][j])
            sub_deter.append(new_row)
    return sub_deter

def calc_deter(the_deter):
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
        sub_det = take_sub_deter(the_deter, 0, j)
        cofactor = the_deter[0][j] * ((-1) ** j)
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
