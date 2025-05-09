"""因子分析及Python应用 Demo"""

import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer as FA
import matplotlib.pyplot as plt
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

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

def Factors(fa):
    return [f'F{i}' for i in range(1, fa.n_factors+1)]

Fp = FA(n_factors=6, method='principal', rotation=None).fit(d71.values)
DF_principal = pd.DataFrame(Fp.loadings_, index=d71.columns, columns=Factors(Fp))
print(DF_principal)

Fp1 = FA(n_factors=3, method='principal', rotation=None).fit(d71.values)
Fp1_load = pd.DataFrame(Fp1.loadings_, index=d71.columns, columns=Factors(Fp1))
print(Fp1_load)

Fm = FA(n_factors=6, method='ml', rotation=None).fit(d71.values)
DF_ml = pd.DataFrame(Fm.loadings_, index=d71.columns, columns=Factors(Fm))
print(DF_ml)

Fm1 = FA(n_factors=3, method='ml', rotation=None).fit(d71.values)
Fm1_load = pd.DataFrame(Fm1.loadings_, index=d71.columns, columns=Factors(Fm1))
print(Fm1_load)

Vars = ['方差', '贡献率', '累计贡献率']

Fp1_Vars = pd.DataFrame(
    [Fp1.get_factor_variance()[i] for i in range(3)],
    index=Vars,
    columns=Factors(Fp1)
)
print(Fp1_Vars)

Fm1_Vars = pd.DataFrame(
    [Fm1.get_factor_variance()[i] for i in range(3)],
    index=Vars,
    columns=Factors(Fm1)
)
print(Fm1_Vars)

Fp1_load['共同度'] = 1 - Fp1.get_uniquenesses()
print(Fp1_load)

Fp2 = FA(n_factors=3, method='principal', rotation='varimax').fit(d71.values)
Fp2_Vars = pd.DataFrame(
    [Fp2.get_factor_variance()[i] for i in range(3)],
    index=Vars,
    columns=Factors(Fp2)
)
print(Fp2_Vars)

Fp2_load = pd.DataFrame(Fp2.loadings_, index=d71.columns, columns=Factors(Fp2))
Fp2_load['共同度'] = 1 - Fp2.get_uniquenesses()
print(Fp2_load)

Fp1_scores = pd.DataFrame(
    Fp1.transform(d71.values),
    index=d71.index,
    columns=Factors(Fp1)
)
print(Fp1_scores.head())

Fp2_scores = pd.DataFrame(
    Fp2.transform(d71.values),
    index=d71.index,
    columns=Factors(Fp2)
)
print(Fp2_scores.head())

def Biplot(Load, Score):
    plt.figure(figsize=(10, 8))
    plt.plot(Score.iloc[:, 0], Score.iloc[:, 1], '*')
    plt.xlabel(Score.columns[0])
    plt.ylabel(Score.columns[1])
    plt.axhline(y=0, ls=':')
    plt.axvline(x=0, ls=':')
    for i in range(len(Load)):
        plt.text(Load.iloc[i, 0], Load.iloc[i, 1], Load.index[i])
    plt.title('因子得分双向图')
    plt.grid(True)
    plt.show()

def FArank(Vars, Scores):
    Vi = Vars.values[0]
    Wi = Vi / sum(Vi)
    Fi = Scores.dot(Wi)
    Ri = Fi.rank(ascending=False).astype(int)
    return pd.DataFrame({'因子得分': Fi, '因子排名': Ri})

# 模拟市场价格数据
categories = ['食品', '衣着', '设备', '医疗', '交通', '教育', '居住', '杂项']
n_regions = 31
base = np.random.randn(n_regions, 1) * 5 + 100
noise_level = 3
d31_data = {}
for cat in categories:
    if cat == '衣着' or cat == '医疗':
        d31_data[cat] = base.flatten() + np.random.randn(n_regions) * noise_level * 1.5
    else:
        d31_data[cat] = base.flatten() + np.random.randn(n_regions) * noise_level
d31 = pd.DataFrame(d31_data)

corr_matrix = d31.corr()
print(corr_matrix)

kmo_all, kmo_model = calculate_kmo(d31)
print(f"KMO: {kmo_model:.4f}")

if kmo_model > 0.6:
    fa_applied = FA(n_factors=2, rotation='varimax').fit(d31)
    fa_loadings = pd.DataFrame(
        fa_applied.loadings_,
        index=d31.columns,
        columns=[f'因子{i+1}' for i in range(fa_applied.n_factors)]
    )
    print(fa_loadings)
    
    fa_vars = pd.DataFrame(
        [fa_applied.get_factor_variance()[i] for i in range(3)],
        index=['方差', '贡献率', '累计贡献率'],
        columns=[f'因子{i+1}' for i in range(fa_applied.n_factors)]
    )
    print(fa_vars)
    
    fa_scores = pd.DataFrame(
        fa_applied.transform(d31),
        index=d31.index,
        columns=[f'因子{i+1}' for i in range(fa_applied.n_factors)]
    )
    print(fa_scores.head())
    
    fa_ranking = FArank(fa_vars, fa_scores)
    print(fa_ranking)

# 因子分析综合评价函数
def FAscores(X, m=2, rot='varimax'):
    import factor_analyzer as fa
    kmo = fa.calculate_kmo(X)
    chisq = fa.calculate_bartlett_sphericity(X)
    print('KMO 检验: KMO 值=%6.4f 卡方值=%8.4f, p 值=%5.4f' % 
          (kmo[1], chisq[0], chisq[1]))
    
    from factor_analyzer import FactorAnalyzer as FA
    Fp = FA(n_factors=m, method='principal', rotation=rot).fit(X.values)
    vars = Fp.get_factor_variance()
    Factor = ['F%d' % (i+1) for i in range(m)]
    Vars = pd.DataFrame(vars, ['方差', '贡献率', '累计贡献率'], Factor)
    print("\n 方差贡献:\n", Vars)
    
    Load = pd.DataFrame(Fp.loadings_, X.columns, Factor)
    Load['共同度'] = 1 - Fp.get_uniquenesses()
    Load = pd.DataFrame(Fp.loadings_, X.columns, Factor)
    Load['共同度'] = 1 - Fp.get_uniquenesses()
    print("\n 因子载荷:\n", Load)
    
    Scores = pd.DataFrame(Fp.transform(X.values), X.index, Factor)
    print("\n 因子得分:\n", Scores)
    
    Vi = vars[0]
    Wi = Vi / sum(Vi)
    Fi = Scores.dot(Wi)
    Ri = Fi.rank(ascending=False).astype(int)
    print("\n 综合排名:\n")
    return pd.DataFrame({'综合得分': Fi, '综合排名': Ri})

# 应用FAscores函数
pd.set_option('display.max_rows', 31)
result = FAscores(d31, m=2, rot='varimax')
print(result)