def find_factors(n):
    factors = []
    for i in range(1, n + 1):
        if n % i == 0:
            factors.append(i)
    return factors

number = int(input("输入一个整数: "))
factors = find_factors(number)
print(f"{number} 的因数有: {factors}")
