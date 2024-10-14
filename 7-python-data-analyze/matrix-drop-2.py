# 定义矩阵 A
A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

def take_sub_deter(local_deter, row_to_delete, column_to_delete):
    """
    生成子行列式，删除指定的行和列
    """
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
    """
    递归计算行列式的值
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
        sub_det = take_sub_deter(the_deter, 0, j)
        cofactor = the_deter[0][j] * ((-1) ** j)
        det_value += cofactor * calc_deter(sub_det)
    
    return det_value

def print_matrix(matrix):
    """
    格式化打印矩阵
    """
    for row in matrix:
        print("[", end=" ")
        for elem in row:
            print(f"{elem:4}", end=" ")
        print("]")

# 主程序
if __name__ == "__main__":
    print("原始矩阵 A:")
    print_matrix(A)
    
    result = calc_deter(A)
    print(f"\n矩阵 A 的行列式为: {result}")
