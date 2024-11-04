
class Matrix:
    def __init__(self, matrix):
        self.matrix = matrix

    def take_sub_deter(self, row_to_delete, column_to_delete):
        sub_deter = []
        for i in range(len(self.matrix)):
            if i != row_to_delete:
                new_row = []
                for j in range(len(self.matrix[i])):
                    if j != column_to_delete:
                        new_row.append(self.matrix[i][j])
                sub_deter.append(new_row)
        return sub_deter

    def calc_deter(self):
        n = len(self.matrix)
        
        # 基本情况：1x1 矩阵
        if n == 1:
            return self.matrix[0][0]
        
        # 基本情况：2x2 矩阵
        if n == 2:
            return self.matrix[0][0] * self.matrix[1][1] - self.matrix[0][1] * self.matrix[1][0]
        
        # 递归情况：nxn 矩阵 (n > 2)
        det_value = 0
        for j in range(n):
            sub_det = self.take_sub_deter(0, j)
            cofactor = self.matrix[0][j] * ((-1) ** j) # 代数余子式
            det_value += cofactor * Matrix(sub_det).calc_deter()
        
        return det_value

    @staticmethod
    def print_matrix(matrix):
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