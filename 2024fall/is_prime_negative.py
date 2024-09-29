import math

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(abs(n))) + 1):
        if n % i == 0:
            return False
    return True

def find_prime_factors(n):
    factors = []
    sign = 1 if n > 0 else -1
    n = abs(n)
    i = 2
    while i * i <= n:
        if n % i == 0 and is_prime(i):
            factors.append(i)
            n //= i
        else:
            i += 1
    if n > 1:
        factors.append(n)
    return [sign] + factors if sign == -1 else factors

def process_number(num):
    if num == 0:
        print("0 既不是素数也不是合数,它没有素因子。")
    elif is_prime(num):
        print(f"{num} 是素数")
    else:
        print(f"{num} 不是素数")
        prime_factors = find_prime_factors(num)
        if num < 0:
            print(f"{num} 的素数因子是: -1 * {prime_factors[1:]}")
        else:
            print(f"{num} 的素数因子是: {prime_factors}")

num = int(input("请输入一个整数: "))
process_number(num)