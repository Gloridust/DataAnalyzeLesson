import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import seaborn as sns

np.random.seed(42)
plt.rcParams['font.family'] = ['STHeiti']

print("=== 多元统计分析演示 ===\n")

print("1. 数据生成")
n = 18
groups = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
y_values = [2.36, 2.38, 2.48, 2.45, 2.47, 2.41, 2.56, 2.58, 2.52, 2.54, 2.59, 2.57, 2.64, 2.59, 2.67, 2.66, 2.62, 2.65]

d111 = pd.DataFrame({
    'Y': y_values,
    'A': groups
})

print("数据预览:")
print(d111.head(10))
print(f"数据形状: {d111.shape}\n")

print("2. 描述性统计分析")
desc_stats = d111.groupby('A')['Y'].agg(['count', 'mean', 'std'])
print("各组统计量:")
print(desc_stats)
print()

print("3. 方差分析 (ANOVA)")
model = ols('Y ~ C(A)', data=d111).fit()
anova_results = anova_lm(model, typ=2)
print("方差分析结果:")
print(anova_results)
print()

print("4. Logistic回归分析")
np.random.seed(123)
n_samples = 45

x1 = np.random.normal(0, 1, n_samples)
x2 = np.random.uniform(0, 10, n_samples)
x3 = np.random.normal(2, 0.5, n_samples)

linear_combination = 0.6 - 1.5*x1 - 0.002*x2 + 0.32*x3
probabilities = 1 / (1 + np.exp(-linear_combination))
y_binary = np.random.binomial(1, probabilities, n_samples)

d115 = pd.DataFrame({
    'y': y_binary,
    'x1': x1,
    'x2': x2,
    'x3': x3
})

print("Logistic回归数据预览:")
print(d115.head())
print()

logit_model = sm.GLM(d115['y'], sm.add_constant(d115[['x1', 'x2', 'x3']]), 
                     family=sm.families.Binomial()).fit()
print("完整Logistic回归结果:")
print(logit_model.summary())
print()

print("5. 变量选择后的简化模型")
logit_x1 = sm.GLM(d115['y'], sm.add_constant(d115[['x1']]), 
                  family=sm.families.Binomial()).fit()
print("简化模型结果:")
print(logit_x1.summary())
print()

print("6. 实际案例分析")
np.random.seed(456)
n_case = 40

age = np.random.randint(18, 75, n_case)
sex = np.random.choice([0, 1], n_case)

linear_comb = 2.92 - 1.07*sex - 0.056*age
prob_case = 1 / (1 + np.exp(-linear_comb))
y_case = np.random.binomial(1, prob_case, n_case)

case11 = pd.DataFrame({
    'y': y_case,
    'sex': sex,
    'age': age
})

print("实际案例数据预览:")
print(case11.head())
print()

glm1 = sm.GLM(case11['y'], sm.add_constant(case11[['sex', 'age']]), 
              family=sm.families.Binomial()).fit()
print("实际案例Logistic回归结果:")
print(glm1.summary())
print()

print("7. 预测概率计算和可视化")
fitted_values = glm1.fittedvalues
predicted_prob = fitted_values

plt.figure(figsize=(10, 6))
plt.scatter(case11['age'], predicted_prob, alpha=0.7, s=50)
plt.xlabel('年龄')
plt.ylabel('预测概率')
plt.title('年龄与预测概率的关系')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("8. 模型评估")
print(f"对数似然值: {glm1.llf:.3f}")
print(f"AIC: {glm1.aic:.3f}")
print(f"偏差: {glm1.deviance:.3f}")

predicted_binary = (predicted_prob > 0.5).astype(int)
accuracy = np.mean(predicted_binary == case11['y'])
print(f"预测准确率: {accuracy:.3f}")

print("\n9. 系数解释")
coefficients = glm1.params
print("模型系数:")
for i, (name, coef) in enumerate(coefficients.items()):
    print(f"{name}: {coef:.4f}")

odds_ratios = np.exp(coefficients)
print("\n优势比 (Odds Ratios):")
for name, or_val in zip(coefficients.index, odds_ratios):
    print(f"{name}: {or_val:.4f}")

print("\n=== 分析完成 ===")