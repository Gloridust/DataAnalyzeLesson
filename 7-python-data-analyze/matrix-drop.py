def calc_deter(matrix):
    # 获取矩阵的阶数
    n = len(matrix)
    
    # 如果是二阶矩阵，直接计算行列式
    if n == 2:
        det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        print("二阶矩阵:")
        for row in matrix:
            print(row)
        print(f"二阶行列式: {det}")
        return det
    
    det = 0
    # 使用第一行元素进行展开
    for j in range(n):
        # 创建子矩阵，删除第一行和第j列
        sub_matrix = [row[:j] + row[j+1:] for row in matrix[1:]]
        # 递归计算子矩阵的行列式，并与代数余子式相乘
        det += (-1) ** j * matrix[0][j] * calc_deter(sub_matrix)
    
    return det

def main():
    # 定义输入矩阵
    matrix = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]
    
    print("原始矩阵:")
    for row in matrix:
        print(row)
    print("\n开始计算行列式:")
    
    # 调用calc_deter函数计算行列式
    result = calc_deter(matrix)
    print(f"\n最终矩阵的行列式为: {result}")

if __name__ == "__main__":
    main()