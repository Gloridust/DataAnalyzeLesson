def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(abs(n)**0.5) + 1):
        if n % i == 0:
            return False
    return True

def find_prime_factors(n):
    n = abs(n)
    i = 2
    factors = []
    while n > 1:
        if n % i == 0 and is_prime(i):
            factors.append(i)
            n //= i
        else:
            i += 1
    return factors

def check_prime(num):
    if num == 0:
        return "0 既不是素数也不是合数"
    
    is_negative = num < 0
    abs_num = abs(num)
    
    if is_prime(abs_num):
        result = f"{num} 是素数" if not is_negative else f"{num} 的绝对值 {abs_num} 是素数"
    else:
        result = f"{num} 不是素数" if not is_negative else f"{num} 的绝对值 {abs_num} 不是素数"
        prime_factors = find_prime_factors(abs_num)
        result += f"\n{abs_num} 的素数因子是: {prime_factors}"
    
    return result

# 主程序
while True:
    try:
        num = int(input("请输入一个整数 (输入 'q' 退出): "))
        print(check_prime(num))
    except ValueError:
        user_input = input("无效输入。是否要退出？(y/n): ")
        if user_input.lower() == 'y':
            break
        else:
            continue