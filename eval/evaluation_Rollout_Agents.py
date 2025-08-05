import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os

# Configurações estéticas
sns.set(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = [14, 10]
plt.rcParams['font.size'] = 12

# Lista manual dos arquivos CSV - TROCAR PARA RANDOM OU CLUSTERING

"""
csv_files = [
    '../logs-Random/active_learning_20250706_135859/progress.csv',
    '../logs-Random/active_learning_20250708_083624/progress.csv',
    '../logs-Random/active_learning_20250710_035320/progress.csv',
    '../logs-Random/active_learning_20250712_170701/progress.csv',
    '../logs-Random/active_learning_20250714_170733/progress.csv',
    '../logs-Random/active_learning_20250716_220745/progress.csv',
    '../logs-Random/active_learning_20250720_210156/progress.csv',
    '../logs-Random/active_learning_20250725_001541/progress.csv',
    '../logs-Random/active_learning_20250729_144025/progress.csv',
    '../logs-Random/active_learning_20250803_030210/progress.csv'
]  # ATUALIZE ESTES CAMINHOS
"""
csv_files = [
    '../logs-clustering/active_learning_20250706_163208/progress.csv',
    '../logs-clustering/active_learning_20250708_110731/progress.csv',
    '../logs-clustering/active_learning_20250710_062425/progress.csv',
    '../logs-clustering/active_learning_20250712_193934/progress.csv',
    '../logs-clustering/active_learning_20250715_025302/progress.csv',
    '../logs-clustering/active_learning_20250717_003937/progress.csv',
    '../logs-clustering/active_learning_20250720_232945/progress.csv',
    '../logs-clustering/active_learning_20250725_152709/progress.csv',
    '../logs-clustering/active_learning_20250729_210258/progress.csv',
    '../logs-clustering/active_learning_20250803_052920/progress.csv'
]  


agents_data = {}

for idx, file in enumerate(csv_files):
    # Extrair nome significativo do caminho do arquivo
    agent_name = f"Agent {idx+1}"  # Ou use: os.path.basename(file).replace('.csv', '')
    try:
        agents_data[agent_name] = pd.read_csv(file)
        agents_data[agent_name]['Agent'] = agent_name
    except Exception as e:
        print(f"Erro ao ler {file}: {str(e)}")
        continue

for idx, file in enumerate(csv_files):
    agent_name = f"Agent {idx+1}"
    agents_data[agent_name] = pd.read_csv(file)
    
    # Adicionar coluna de identificação do agente
    agents_data[agent_name]['Agent'] = agent_name

# 2. Combinar todos os dados em um único DataFrame
combined_df = pd.concat(agents_data.values())

# 3. Pré-processamento
# Filtrar métricas de Rollout
rollout_metrics = ['time/iterations', 'rollout/ep_rew_mean', 'rollout/ep_len_mean', 'Agent']
rollout_df = combined_df[rollout_metrics].copy()

# Remover linhas com valores ausentes
rollout_df.dropna(inplace=True)

# 4. Visualização comparativa
fig, ax = plt.subplots(2, 1, figsize=(14, 12))

# Gráfico 1: Recompensa Média Comparada
sns.lineplot(
    x='time/iterations',
    y='rollout/ep_rew_mean',
    hue='Agent',
    data=rollout_df,
    ax=ax[0],
    errorbar=('ci', 95),  # Intervalo de confiança de 95%
    linewidth=2
)
ax[0].set_title('Comparison of average reward by episode (rollout)', fontsize=16, pad=15)
ax[0].set_ylabel('Average reward', fontsize=14)
ax[0].set_xlabel('Training iterations', fontsize=14)
ax[0].legend(title='Agents', title_fontsize=13, loc='lower right')

# Gráfico 2: Duração Média Comparada
sns.lineplot(
    x='time/iterations',
    y='rollout/ep_len_mean',
    hue='Agent',
    data=rollout_df,
    ax=ax[1],
    errorbar=('ci', 95),
    linewidth=2
)
ax[1].set_title('Comparison of average duration of the episodes (rollout)', fontsize=16, pad=15)
ax[1].set_ylabel('Steps by episode', fontsize=14)
ax[1].set_xlabel('Training iterations', fontsize=14)
ax[1].legend(title='Agents', title_fontsize=13, loc='lower right')

# 5. Análise de desempenho final (últimas iterações)
plt.figure(figsize=(12, 8))
last_iterations = rollout_df.groupby('Agent').apply(lambda x: x.nlargest(20, 'time/iterations'))
sns.boxplot(
    x='Agent',
    y='rollout/ep_rew_mean',
    data=last_iterations,
    showmeans=True,
    meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black"}
)
plt.title('Final performance (last 20 iterations)', fontsize=16)
plt.ylabel('Average reward', fontsize=14)
plt.xlabel('Agent', fontsize=14)
plt.xticks(rotation=15)
plt.grid(axis='y', alpha=0.3)

# 6. Salvar resultados
plt.tight_layout()
plt.savefig('comparacao_agentes.png', dpi=300)
plt.show()