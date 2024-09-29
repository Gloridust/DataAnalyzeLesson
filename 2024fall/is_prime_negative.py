def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def find_prime_factors(n):
    n = abs(n)  # 处理负数
    i = 2
    factors = []
    while n > 1:
        if n % i == 0 and is_prime(i):
            factors.append(i)
            n //= i
        else:
            i += 1
    return factors

def find_factors(n):
    n = abs(n)  # 处理负数
    factors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            factors.append(i)
            if i != n // i:
                factors.append(n // i)
    return sorted(factors)

def analyze_number(n):
    if n == 0:
        return "0 既不是素数也不是合数,它没有因子。"
    
    abs_n = abs(n)
    
    if is_prime(abs_n):
        return f"{n} 是{'负' if n < 0 else '正'}素数"
    else:
        factors = find_factors(n)
        prime_factors = find_prime_factors(n)
        return f"{n} 不是素数\n" \
               f"{n} 的所有因子是: {factors}\n" \
               f"{n} 的素数因子是: {prime_factors}"

# 主程序
while True:
    try:
        num = int(input("请输入一个整数 (输入0退出): "))
        if num == 0:
            print("程序结束。")
            break
        result = analyze_number(num)
        print(result)
    except ValueError:
        print("请输入有效的整数。")