import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 14
rcParams['axes.titleweight'] = 'bold'

file_path = 'IA_models.csv'
data = pd.read_csv(file_path)

data['Publication date'] = pd.to_datetime(data['Publication date'], errors='coerce')
data = data.dropna(subset=['Publication date'])

models_by_date = data.groupby(data['Publication date'].dt.to_period('M')).size()

start = models_by_date.index.min().to_timestamp()
end = models_by_date.index.max().to_timestamp()

full_index = pd.date_range(start=start, end=end, freq='MS')
models_by_date = models_by_date.to_timestamp().reindex(full_index, fill_value=0)

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(models_by_date.index, models_by_date.values, color='darkorange', linewidth=4)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
    spine.set_color("#888888")

ax.set_xlim(models_by_date.index.min(), models_by_date.index.max())

xticks = [i for i in models_by_date.index if i.month == 1]
xticks_labels = [str(i.year) for i in xticks]

ax.set_xticks(xticks)
ax.set_xticklabels(xticks_labels, rotation=0)

ax.set_title('Modelos de IA por Ano de Publicação', fontsize=20, pad=15)
ax.set_xlabel('Ano', fontsize=14)
ax.set_ylabel('Número de Modelos', fontsize=14)

plt.tight_layout()
plt.show()