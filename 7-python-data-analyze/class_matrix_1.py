class Matrix:
    def __init__(self, matrix):
        # 初始化方法，用于创建类的实例时传入矩阵数据
        self.matrix = matrix
    @staticmethod
    def calc_determinant(matrix):
        # 计算给定矩阵的行列式
        n = len(matrix)
        
        # 基本情况：1x1 矩阵
        if n == 1:
            return matrix[0][0]
        
        # 基本情况：2x2 矩阵
        elif n == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        
        # 递归情况：nxn 矩阵 (n > 2)
        sum = 0
        for column_num in range(len(matrix[0])):
            temp_sub_deter = Matrix(Matrix.take_sub_deter_static(matrix, 0, column_num))
            sum = (sum + 
                  matrix[0][column_num] * ((-1) ** (1 + column_num + 1)) 
                  * temp_sub_deter.calc_deter())
        
        return sum

    @staticmethod
    def take_sub_deter_static(matrix, row_to_delete, column_to_delete):
        # 从给定矩阵中删除指定的行和列，返回子矩阵
        sub_deter = []
        for i in range(len(matrix)):
            if i != row_to_delete:
                new_row = []
                for j in range(len(matrix[i])):
                    if j != column_to_delete:
                        new_row.append(matrix[i][j])
                sub_deter.append(new_row)
        return sub_deter

    def calc_deter(self):
        # 计算当前实例矩阵的行列式
        return Matrix.calc_determinant(self.matrix)

    @staticmethod
    def print_matrix(matrix):
        # 打印矩阵的静态方法
        for row in matrix:
            print("[", end=" ")
            for elem in row:
                print(f"{elem:4}", end=" ")
            print("]")

# 主程序
if __name__ == "__main__":
    # 定义矩阵 A
    A = [[1, 2, 2, 3], 
        [4, 5, 6, 6],
        [7, 8, 8, 9],
        [7, 5, 9, 6]]
    
    matrix_instance = Matrix(A)
    print("原始矩阵 A:")
    Matrix.print_matrix(A)
    
    result = matrix_instance.calc_deter()
    print(f"\n矩阵 A 的行列式为: {result}")