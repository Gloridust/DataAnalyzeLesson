# 面向对象练习

class Student:
    def __init__(self, name, score):
        self.name = name
        self.score = score
    
    def print_score(self):
        print('{0},{1}'.format(self.name, self.score))


bart = Student('Bart Simpson', 59)
lisa = Student('Lisa Simpson', 87)

bart.print_score()
lisa.print_score()
