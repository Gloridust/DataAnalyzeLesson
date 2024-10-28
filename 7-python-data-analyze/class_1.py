# 面向对象练习

class Student:
    def __init__(self, name, score, gender):
        self.name = name
        self.score = score
        self.gender = gender
    
    def print_score(self):
        print('{0},{1},{2}'.format(self.name, self.score, self.gender))


bart = Student('Bart Simpson', 59, 'Male')
lisa = Student('Lisa Simpson', 87, 'Female')
alex = Student('Alex Johnson', 75, 'Gay')
taylor = Student('Taylor Smith', 82, 'Transgender')

bart.print_score()
lisa.print_score()
alex.print_score()
taylor.print_score()

# 这个文件展示了一个简单的面向对象编程示例。我们定义了一个名为 Student 的类，
# 该类有两个属性：name 和 score。类中有一个方法 print_score，用于打印学生的姓名和分数。
# 然后，我们创建了两个 Student 对象，分别是 Bart Simpson 和 Lisa Simpson，并调用了 print_score 方法来打印他们的分数。
