import numpy as np

class ArrayOperations:
    def __init__(self):
        # 创建不同类型的数组
        self.a = np.array([1, 2, 3, 4])
        self.b = np.array((5, 6, 7, 8))
        self.c = np.array([[1, 2, 3, 4],
                          [4, 5, 6, 7], 
                          [7, 8, 9, 10]])
        
    def print_array_info(self, arr, name):
        print(f"\n{name} 数组信息:")
        print(f"数据: \n{arr}")
        print(f"数据类型: {arr.dtype}")
        print(f"形状: {arr.shape}")
        print(f"大小: {arr.size}")
        
    def demonstrate_array_types(self):
        # 使用不同的数据类型创建数组
        float_arr = np.array(self.c, dtype=np.float32)
        complex_arr = np.array(self.c, dtype=np.complex128)
        
        print("\n不同数据类型的数组:")
        print("Float32类型:")
        print(float_arr)
        print("\nComplex类型:")
        print(complex_arr)
        
    def demonstrate_array_functions(self):
        print("\n常用数组函数:")
        print("arange(0, 10, 2):", np.arange(0, 10, 2))  # 创建等差数列
        print("linspace(0, 1, 5):", np.linspace(0, 1, 5))  # 创建等间隔数列
        print("logspace(0, 2, 3):", np.logspace(0, 2, 3))  # 创建等比数列
        print("zeros((2,2)):\n", np.zeros((2,2)))  # 创建全0数组
        print("ones((2,2)):\n", np.ones((2,2)))  # 创建全1数组

def main():
    ops = ArrayOperations()
    
    # 打印各个数组的信息
    ops.print_array_info(ops.a, "a")
    ops.print_array_info(ops.b, "b")
    ops.print_array_info(ops.c, "c")
    
    # 演示不同数据类型
    ops.demonstrate_array_types()
    
    # 演示常用数组函数
    ops.demonstrate_array_functions()

if __name__ == "__main__":
    main()
