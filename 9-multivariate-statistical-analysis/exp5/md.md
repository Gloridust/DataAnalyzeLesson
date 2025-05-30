# 因子分析及Python应用实验报告

## 实验目的

本次实验旨在掌握多元统计分析中因子分析的基本原理与方法，通过Python编程实现主因子法、极大似然法等不同因子提取方法，理解因子载荷、因子旋转、因子得分等概念，并学习如何应用因子分析解决实际问题，进行数据降维和综合评价。

## 实验准备

在实验前，我安装了必要的Python库包括pandas、numpy、matplotlib和factor_analyzer，这些工具能够帮助实现因子分析的各项功能。我还预习了因子分析的相关理论知识，包括因子模型的建立、因子载荷的计算与解释、因子旋转的方法以及因子得分的计算等内容，为实验的顺利进行做好了准备。

```python
import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer as FA
import matplotlib.pyplot as plt
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
```

## 实验内容

### 1. 数据准备与模型构建

我首先创建了模拟数据集作为分析对象，这个数据集包含6个变量，潜在的三个因子结构。通过设定合理的因子载荷矩阵和添加适当的噪声，构建了一个适合进行因子分析的数据集。

```python
# 创建模拟数据
np.random.seed(42)
n_samples = 200
n_features = 6

latent_factors = np.random.randn(n_samples, 3)
noise = np.random.randn(n_samples, n_features) * 0.3

loadings = np.array([
    [0.7, 0.5, -0.3],  # x1
    [0.8, 0.4, -0.3],  # x2
    [-0.5, 0.7, 0.1],  # x3
    [0.6, -0.7, -0.2], # x4
    [0.6, -0.1, 0.7],  # x5
    [0.5, 0.3, 0.7]    # x6
])

d71_values = np.dot(latent_factors, loadings.T) + noise
d71 = pd.DataFrame(d71_values, columns=[f'x{i+1}' for i in range(n_features)])
```

### 2. 主因子法分析

接着我使用主因子法(Principal Component Method)对数据进行了分析。首先尝试提取全部6个因子，结果显示各变量在不同因子上的载荷分布明显。

```python
# 使用主成分法提取6个因子
Fp = FA(n_factors=6, method='principal', rotation=None).fit(d71.values)
DF_principal = pd.DataFrame(Fp.loadings_, index=d71.columns, columns=Factors(Fp))
```

运行结果：
```
          F1        F2        F3        F4        F5        F6
x1  0.636592  0.560656 -0.480330  0.143895 -0.059413  0.159571
x2  0.691961  0.492628 -0.472885 -0.195057  0.030244 -0.126197
x3 -0.505837  0.817574  0.164415  0.034975  0.216963  0.019323
x4  0.581119 -0.717709 -0.310526  0.106982  0.196557 -0.026237
x5  0.683862 -0.098113  0.689230 -0.145819  0.054798  0.152977
x6  0.632430  0.316147  0.672442  0.155925 -0.039618 -0.148400
```

随后我减少了因子数量，提取了前3个主因子，发现这三个因子已经能够解释数据中大部分的变异。

```python
# 提取3个主因子
Fp1 = FA(n_factors=3, method='principal', rotation=None).fit(d71.values)
Fp1_load = pd.DataFrame(Fp1.loadings_, index=d71.columns, columns=Factors(Fp1))
```

运行结果：
```
          F1        F2        F3
x1  0.636592  0.560656 -0.480330
x2  0.691961  0.492628 -0.472885
x3 -0.505837  0.817574  0.164415
x4  0.581119 -0.717709 -0.310526
x5  0.683862 -0.098113  0.689230
x6  0.632430  0.316147  0.672442
```

从主因子法的方差贡献率来看，第一个因子解释了约39.1%的总方差，累计三个因子能解释约95.0%的总方差。

```python
# 计算方差贡献率
Vars = ['方差', '贡献率', '累计贡献率']
Fp1_Vars = pd.DataFrame(
    [Fp1.get_factor_variance()[i] for i in range(3)],
    index=Vars,
    columns=Factors(Fp1)
)
```

运行结果：
```
             F1        F2        F3
方差     2.345265  1.850126  1.505012
贡献率    0.390877  0.308354  0.250835
累计贡献率  0.390877  0.699232  0.950067
```

### 3. 极大似然法分析

我还尝试了极大似然法(Maximum Likelihood Method)进行因子分析，并与主因子法的结果进行了比较。

```python
# 使用极大似然法提取3个因子
Fm1 = FA(n_factors=3, method='ml', rotation=None).fit(d71.values)
Fm1_load = pd.DataFrame(Fm1.loadings_, index=d71.columns, columns=Factors(Fm1))
```

运行结果：
```
          F1        F2        F3
x1  0.078479  0.987717  0.006151
x2  0.137412  0.894389 -0.058794
x3 -0.319884  0.073429  0.892352
x4  0.267044  0.102332 -0.905436
x5  0.996864 -0.029258  0.020142
x6  0.819009  0.193702  0.325396
```

极大似然法得到的三个因子累计贡献率：

```python
# 极大似然法方差贡献
Fm1_Vars = pd.DataFrame(
    [Fm1.get_factor_variance()[i] for i in range(3)],
    index=Vars,
    columns=Factors(Fm1)
)
```

运行结果：
```
             F1        F2        F3
方差     1.863194  1.829757  1.725890
贡献率    0.310532  0.304959  0.287648
累计贡献率  0.310532  0.615492  0.903140
```

### 4. 共同度分析

在计算共同度时，发现所有变量的共同度都非常高，都超过了0.94，说明提取的因子能够很好地解释原始变量的变异性。

```python
# 计算共同度
Fp1_load['共同度'] = 1 - Fp1.get_uniquenesses()
```

运行结果：
```
          F1        F2        F3       共同度
x1  0.636592  0.560656 -0.480330  0.950301
x2  0.691961  0.492628 -0.472885  0.945112
x3 -0.505837  0.817574  0.164415  0.951330
x4  0.581119 -0.717709 -0.310526  0.949232
x5  0.683862 -0.098113  0.689230  0.952332
x6  0.632430  0.316147  0.672442  0.952095
```

### 5. 因子旋转

为了更清晰地解释各因子的含义，我对因子进行了正交旋转(Varimax方法)。

```python
# 使用正交旋转varimax
Fp2 = FA(n_factors=3, method='principal', rotation='varimax').fit(d71.values)
Fp2_load = pd.DataFrame(Fp2.loadings_, index=d71.columns, columns=Factors(Fp2))
Fp2_load['共同度'] = 1 - Fp2.get_uniquenesses()
```

旋转后的结果：
```
          F1        F2        F3       共同度
x1  0.972338 -0.008037  0.069249  0.950301
x2  0.964116  0.073743  0.100767  0.945112
x3  0.056855 -0.970233 -0.082127  0.951330
x4  0.126319  0.965771  0.023679  0.949232
x5 -0.014657  0.241531  0.945399  0.952332
x6  0.195628 -0.119453  0.948449  0.952095
```

旋转后的方差贡献率：
```
             F1        F2        F3
方差     1.932636  1.952176  1.815591
贡献率    0.322106  0.325363  0.302599
累计贡献率  0.322106  0.647469  0.950067
```

### 6. 因子得分计算

通过计算因子得分，我能够对每个样本在各因子上的表现进行评价。

```python
# 计算旋转后因子得分
Fp2_scores = pd.DataFrame(
    Fp2.transform(d71.values),
    index=d71.index,
    columns=Factors(Fp2)
)
```

运行结果：
```
         F1        F2        F3
0  0.067294  0.103215  1.096287
1  0.769358  1.493771  0.597013
2  1.683444 -0.209351  0.603399
3  0.415636  0.810124  0.299852
4  0.064235  1.967670 -1.454706
```

### 7. 实际数据应用

我还针对一组消费类别数据进行了因子分析应用。首先计算了相关系数矩阵：

```
          食品        衣着        设备        医疗        交通        教育        居住        杂项
食品  1.000000  0.731435  0.795246  0.603120  0.760643  0.843130  0.664492  0.836732
衣着  0.731435  1.000000  0.742953  0.627360  0.777418  0.719507  0.683236  0.733859
设备  0.795246  0.742953  1.000000  0.650833  0.711909  0.790638  0.761622  0.675529
医疗  0.603120  0.627360  0.650833  1.000000  0.602194  0.687476  0.585195  0.655426
交通  0.760643  0.777418  0.711909  0.602194  1.000000  0.794849  0.584038  0.785961
教育  0.843130  0.719507  0.790638  0.687476  0.794849  1.000000  0.606738  0.840785
居住  0.664492  0.683236  0.761622  0.585195  0.584038  0.606738  1.000000  0.644892
杂项  0.836732  0.733859  0.675529  0.655426  0.785961  0.840785  0.644892  1.000000
```

在处理前，我首先进行了KMO检验，得到KMO值：
```
KMO: 0.9015
```

这一KMO值远高于通常认为适合因子分析的0.6的标准，表明该数据极其适合进行因子分析。

使用因子分析综合评价函数对消费类别数据进行分析：

```python
def FAscores(X, m=2, rot='varimax'):
    # KMO和Bartlett球形检验
    kmo = calculate_kmo(X)
    chisq = calculate_bartlett_sphericity(X)
    print('KMO 检验: KMO 值=%6.4f 卡方值=%8.4f, p 值=%5.4f' % 
          (kmo[1], chisq[0], chisq[1]))
    
    # 因子分析
    Fp = FA(n_factors=m, method='principal', rotation=rot).fit(X.values)
    vars = Fp.get_factor_variance()
    Factor = ['F%d' % (i+1) for i in range(m)]
    
    # 方差贡献
    Vars = pd.DataFrame(vars, ['方差', '贡献率', '累计贡献率'], Factor)
    print("\n 方差贡献:\n", Vars)
    
    # 因子载荷
    Load = pd.DataFrame(Fp.loadings_, X.columns, Factor)
    Load['共同度'] = 1 - Fp.get_uniquenesses()
    print("\n 因子载荷:\n", Load)
    
    # 因子得分
    Scores = pd.DataFrame(Fp.transform(X.values), X.index, Factor)
    print("\n 因子得分:\n", Scores)
    
    # 综合得分和排名
    Vi = vars[0]
    Wi = Vi / sum(Vi)
    Fi = Scores.dot(Wi)
    Ri = Fi.rank(ascending=False).astype(int)
    print("\n 综合排名:\n")
    return pd.DataFrame({'综合得分': Fi, '综合排名': Ri})
```

运行FAscores函数的结果显示：
```
KMO 检验: KMO 值=0.9015 卡方值=214.3033, p 值=0.0000

 方差贡献:
              F1        F2
方差     3.810004  2.714925
贡献率    0.476251  0.339366
累计贡献率  0.476251  0.815616

 因子载荷:
           F1        F2       共同度
食品  0.790420  0.463513  0.839608
衣着  0.648251  0.581679  0.758581
设备  0.544932  0.734169  0.835955
医疗  0.503987  0.609809  0.625870
交通  0.842799  0.347617  0.831147
教育  0.843107  0.409404  0.878441
居住  0.288885  0.902785  0.898475
杂项  0.842297  0.383912  0.856852
```

在旋转后的因子载荷矩阵中，第一个因子主要由食品(0.79)、交通(0.84)、教育(0.84)和杂项(0.84)这些变量构成，可以解释为"基本生活保障"因子；第二个因子主要由居住(0.90)和设备(0.73)构成，可以解释为"生活质量"因子。

基于因子得分的综合排名结果：
```
        综合得分  综合排名
0   1.652576     1
1  -0.076467    18
2  -0.238598    22
3  -1.042872    29
4  -0.859121    27
...
30  0.624684     6
```

## 实验总结

通过本次实验，我深入理解了因子分析的基本原理和应用方法。因子分析作为一种重要的数据降维和变量归纳技术，能够有效地从众多相关变量中提取少数几个潜在因子，帮助我们揭示数据的内在结构和规律。
在实验中，我掌握了主因子法和极大似然法两种常用的因子提取方法，了解了它们的优缺点和适用场景。我还学习了正交旋转的方法，以及如何通过旋转后的因子载荷矩阵来解释因子的实际含义。通过计算因子得分，我能够对样本进行综合评价和排名。
实验中的KMO检验、因子提取、因子旋转和因子得分计算等步骤，都能通过简洁的Python代码实现。
