import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import rcParams

plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 14
rcParams['axes.titleweight'] = 'bold'

file_path = 'IA_models.csv'
data = pd.read_csv(file_path)

data['Publication date'] = pd.to_datetime(data['Publication date'], errors='coerce')
data = data.dropna(subset=['Publication date'])

models_by_month = data.groupby(data['Publication date'].dt.to_period('M')).size()
models_by_month = models_by_month.to_timestamp()

decomp = seasonal_decompose(models_by_month, model='additive', period=12)

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(
    decomp.trend.index,
    decomp.trend.values,
    color='darkorange',
    linewidth=4,
    label='Tendência'
)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
    spine.set_color("#888888")

ax.set_xlim(decomp.trend.dropna().index.min(), decomp.trend.dropna().index.max())

ax.set_title('Componente de Tendência', fontsize=20, pad=15)
ax.set_xlabel('Ano e mês', fontsize=14)
ax.set_ylabel('Nível de Tendência', fontsize=14)

ax.legend(loc='upper left', fontsize=13)

plt.tight_layout()
plt.show()