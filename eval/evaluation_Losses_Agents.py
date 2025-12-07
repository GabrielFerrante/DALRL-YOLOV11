import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os

# Configurações de estilo
sns.set(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams['figure.figsize'] = [16, 12]
plt.rcParams['font.size'] = 13

# 1. Carregar todos os CSVs - ATUALIZE COM SEUS ARQUIVOS
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

# Se estiverem em outro diretório:
# csv_files = glob.glob('caminho/para/pasta/*.csv')[:10]

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


# 2. Combinar dados
combined_df = pd.concat(agents_data.values())

# 3. Lista de métricas de treinamento importantes
train_metrics = [
    'train/value_loss',
    'train/entropy_loss',
    'train/loss',
    'train/approx_kl',
    'train/clip_fraction',
    'train/policy_gradient_loss',
    'train/explained_variance',
    'train/learning_rate',
    'time/iterations',
    'Agent',
    'rollout/ep_rew_mean',  # Recompensa média por episódio
]

# 4. Filtrar e pré-processar
train_df = combined_df[train_metrics].copy()
train_df.dropna(inplace=True) # Mantém linhas com pelo menos 1 métrica

# 5. Criar visualizações
# =============================================================================
# Gráfico 1: Métricas de Loss (4 principais)
# =============================================================================

# Definir uma lista de 10 marcadores geométricos diferentes
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'X']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('LOSS metrics during training', fontsize=20)

# Value Loss
sns.lineplot(
    x='time/iterations',
    y='train/value_loss',
    hue='Agent',
     style='Agent',
    data=train_df,
    ax=axes[0, 0],
    errorbar=None,
    linewidth=1.5,
    markers=markers,  # Usar os 10 marcadores diferentes
    dashes=False,  # Remove padrões de linha, mantendo apenas marcadores
)
axes[0, 0].set_title('Loss of value function')
axes[0, 0].set_ylabel('Value Loss')
axes[0, 0].set_xlabel('Iterations')

# Entropy Loss
sns.lineplot(
    x='time/iterations',
    y='train/entropy_loss',
    hue='Agent',
    style='Agent',
    data=train_df,
    ax=axes[0, 1],
    errorbar=None,
    linewidth=1.5,
    markers=markers,  # Usar os 10 marcadores diferentes
    dashes=False,  # Remove padrões de linha, mantendo apenas marcadores
)
axes[0, 1].set_title('Entropy loss')
axes[0, 1].set_ylabel('Entropy Loss')
axes[0, 1].set_xlabel('Iterations')

# Total Loss
sns.lineplot(
    x='time/iterations',
    y='train/loss',
    hue='Agent',
    style='Agent',
    data=train_df,
    ax=axes[1, 0],
    errorbar=None,
    linewidth=1.5,
    markers=markers,  # Usar os 10 marcadores diferentes
    dashes=False,  # Remove padrões de linha, mantendo apenas marcadores
)
axes[1, 0].set_title('Total Loss')
axes[1, 0].set_ylabel('Total Loss')
axes[1, 0].set_xlabel('Iterations')

# Policy Gradient Loss
sns.lineplot(
    x='time/iterations',
    y='train/policy_gradient_loss',
    hue='Agent',
    style='Agent',
    data=train_df,
    ax=axes[1, 1],
    errorbar=None,
    linewidth=1.5,
    markers=markers,  # Usar os 10 marcadores diferentes
    dashes=False,  # Remove padrões de linha, mantendo apenas marcadores
)
axes[1, 1].set_title('Loss of Politics Gradient')
axes[1, 1].set_ylabel('Policy Loss')
axes[1, 1].set_xlabel('Iterations')

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('training_losses_comparison.png', dpi=300)

# =============================================================================
# Gráfico 2: Métricas de Estabilidade e Desempenho
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Stability and performance metrics', fontsize=20)

# Explained Variance
sns.lineplot(
    x='time/iterations',
    y='train/explained_variance',
    hue='Agent',
    style='Agent',
    data=train_df,
    ax=axes[0, 0],
    errorbar=None,
    linewidth=1.5,
    markers=markers,  # Usar os 10 marcadores diferentes
    dashes=False,  # Remove padrões de linha, mantendo apenas marcadores
)
axes[0, 0].set_title('Explained Variance')
axes[0, 0].set_ylabel('Explained Variance')
axes[0, 0].set_ylim(-0.1, 1.1)
axes[0, 0].set_xlabel('Iterations')


# Approx KL
sns.lineplot(
    x='time/iterations',
    y='train/approx_kl',
    hue='Agent',
    style='Agent',  # Isso criará linhas com estilos diferentes
    data=train_df,
    ax=axes[0, 1],
    errorbar=None,
    linewidth=1.5,
    markers=markers,  # Usar os 10 marcadores diferentes
    dashes=False,  # Remove padrões de linha, mantendo apenas marcadores
)
axes[0, 1].set_title('Approximate KL divergence')
axes[0, 1].set_ylabel('KL Divergence')
axes[0, 1].set_yscale('log')
axes[0, 1].set_xlabel('Iterations')

# Clip Fraction
sns.lineplot(
    x='time/iterations',
    y='train/clip_fraction',
    hue='Agent',
    style='Agent',
    data=train_df,
    ax=axes[1, 0],
    errorbar=None,
    linewidth=1.5,
    markers=markers,  # Usar os 10 marcadores diferentes
    dashes=False,  # Remove padrões de linha, mantendo apenas marcadores
)
axes[1, 0].set_title('Clip Fraction')
axes[1, 0].set_ylabel('Clip Fraction')
axes[1, 0].set_ylim(0, 1)
axes[1, 0].set_xlabel('Iterations')

# Learning Rate
sns.lineplot(
    x='time/iterations',
    y='train/learning_rate',
    hue='Agent',
    style='Agent',
    data=train_df,
    ax=axes[1, 1],
    errorbar=None,
    linewidth=1.5,
    markers=markers,  # Usar os 10 marcadores diferentes
    dashes=False,  # Remove padrões de linha, mantendo apenas marcadores
)
axes[1, 1].set_title('Learning Rate')
axes[1, 1].set_ylabel('Learning Rate')
axes[1, 1].set_xlabel('Iterations')

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('training_stability_metrics.png', dpi=300)

# =============================================================================
# Análise de Correlação
# =============================================================================
plt.figure(figsize=(14, 10))

# Selecionar últimas iterações para análise
last_iterations = train_df.groupby('Agent').apply(lambda x: x.nlargest(50, 'time/iterations'))

# Calcular correlação entre value loss e recompensa (se disponível)

print(last_iterations.columns)
if 'rollout/ep_rew_mean' in combined_df.columns and 'train/value_loss' in combined_df.columns:
    corr_df = last_iterations[['train/value_loss', 'rollout/ep_rew_mean']].corr()
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation between Value Loss and average reward (last 50 iterations)', pad=20)
    plt.savefig('value_loss_reward_correlation.png', dpi=300)

plt.show()

print("Análise concluída! Gráficos salvos no diretório atual.")