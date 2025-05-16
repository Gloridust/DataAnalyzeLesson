# 导入所需库
import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 设置中文字体支持
try:
    # 尝试使用 SimHei（黑体）
    font = FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc')
except:
    try:
        # 尝试使用思源黑体（如果已安装）
        font = FontProperties(fname='/System/Library/Fonts/SourceHanSansSC-Regular.otf')
    except:
        try:
            # 尝试使用微软雅黑（如果已安装）
            font = FontProperties(fname='/Library/Fonts/Microsoft/Microsoft YaHei.ttf')
        except:
            # 最后尝试使用任何可用的中文字体
            font = FontProperties(family='sans-serif')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
# 假设你的数据已经准备好，命名为d71
# 如果没有数据，创建一个示例数据
np.random.seed(42)
n_samples = 200
n_features = 6
d71 = pd.DataFrame(
    np.random.randn(n_samples, n_features),
    columns=[f'x{i+1}' for i in range(n_features)]
)

# 定义因子名称函数
def Factors(fa):
    return ['F'+str(i) for i in range(1, fa.n_factors+1)]

# 主因子法分析
from factor_analyzer import FactorAnalyzer as FA
Fp = FA(n_factors=6, method='principal', rotation=None).fit(d71.values)
# 显示因子载荷
print("主因子法载荷矩阵:")
print(pd.DataFrame(Fp.loadings_, index=d71.columns, columns=Factors(Fp)))

# 极大似然法分析
Fm = FA(n_factors=6, method='ml', rotation=None).fit(d71.values)
print("\n极大似然法载荷矩阵:")
print(pd.DataFrame(Fm.loadings_, index=d71.columns, columns=Factors(Fm)))

# 因子方差和共同度
print("\n因子方差:")
print(Fp.get_factor_variance())
print("\n共同度:")
print(Fp.get_communalities())

# 带旋转的因子分析（最大方差法varimax）
Fr = FA(n_factors=3, rotation='varimax', method='minres').fit(d71.values)
print("\n旋转后的因子载荷矩阵:")
print(pd.DataFrame(Fr.loadings_, index=d71.columns, columns=Factors(Fr)))

# 计算因子得分
factor_scores = Fp.transform(d71.values)
print("\n前5行因子得分:")
print(pd.DataFrame(factor_scores[:5], columns=Factors(Fp)))

# 因子得分可视化
plt.figure(figsize=(10, 6))
plt.scatter(factor_scores[:, 0], factor_scores[:, 1])
plt.xlabel('Factor 1', fontproperties=font)
plt.ylabel('Factor 2', fontproperties=font)
plt.title('因子得分分布图 (F1 vs F2)', fontproperties=font)
plt.grid(True)
plt.tight_layout()
plt.savefig('factor_scores.png')  # 保存图片到文件
plt.show()