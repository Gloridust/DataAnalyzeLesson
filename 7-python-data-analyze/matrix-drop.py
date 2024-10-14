import numpy as np

def calc_deter(matrix):
    n = len(matrix)
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    det = 0
    for j in range(n):
        sub_matrix = np.delete(np.delete(matrix, 0, axis=0), j, axis=1)
        det += ((-1) ** j) * matrix[0][j] * calc_deter(sub_matrix)
    
    return det

def main():
    # 定义输入矩阵 (这里以 4x4 矩阵为例)
    matrix = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])
    
    print("输入矩阵:")
    print(matrix)
    
    n = len(matrix)
    result = calc_deter(matrix)
    
    print(f"\n{n}阶矩阵的行列式值为: {result}")

if __name__ == "__main__":
    main()
