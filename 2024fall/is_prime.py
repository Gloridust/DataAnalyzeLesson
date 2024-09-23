nums = []

def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def find_prime_factors(n):
    i = 2
    factors = []
    while n > 1:
        if n % i == 0 and is_prime(i):
            factors.append(i)
            n //= i
        else:
            i += 1
    return factors

num = int(input("请输入一个正整数: "))

if is_prime(num):
    print(f"{num} 是素数")
else:
    print(f"{num} 不是素数")
    prime_factors = find_prime_factors(n)
    print(f"{num} 的素数因子是: {prime_factors}")