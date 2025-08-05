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
"""
# Se estiverem em outro diretório:
# csv_files = glob.glob('caminho/para/pasta/*.csv')[:10]

agents_data = {}
for file in csv_files:
    try:
        agent_name = os.path.splitext(os.path.basename(file))[0]
        df = pd.read_csv(file)
        df['Agent'] = agent_name
        agents_data[agent_name] = df
        print(f"Arquivo {file} carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar {file}: {str(e)}")

if not agents_data:
    print("Nenhum arquivo carregado. Verifique os caminhos.")
    exit()

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
train_df.dropna(subset=train_metrics[:-2], how='all', inplace=True)  # Mantém linhas com pelo menos 1 métrica

# 5. Criar visualizações
# =============================================================================
# Gráfico 1: Métricas de Loss (4 principais)
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Métricas de Loss Durante o Treinamento', fontsize=20)

# Value Loss
sns.lineplot(
    x='time/iterations',
    y='train/value_loss',
    hue='Agent',
    data=train_df,
    ax=axes[0, 0],
    errorbar=None,
    linewidth=1.5
)
axes[0, 0].set_title('Perda da Função de Valor')
axes[0, 0].set_ylabel('Value Loss')
axes[0, 0].set_xlabel('Iterações')

# Entropy Loss
sns.lineplot(
    x='time/iterations',
    y='train/entropy_loss',
    hue='Agent',
    data=train_df,
    ax=axes[0, 1],
    errorbar=None,
    linewidth=1.5
)
axes[0, 1].set_title('Perda de Entropia')
axes[0, 1].set_ylabel('Entropy Loss')
axes[0, 1].set_xlabel('Iterações')

# Total Loss
sns.lineplot(
    x='time/iterations',
    y='train/loss',
    hue='Agent',
    data=train_df,
    ax=axes[1, 0],
    errorbar=None,
    linewidth=1.5
)
axes[1, 0].set_title('Perda Total')
axes[1, 0].set_ylabel('Total Loss')
axes[1, 0].set_xlabel('Iterações')

# Policy Gradient Loss
sns.lineplot(
    x='time/iterations',
    y='train/policy_gradient_loss',
    hue='Agent',
    data=train_df,
    ax=axes[1, 1],
    errorbar=None,
    linewidth=1.5
)
axes[1, 1].set_title('Perda do Gradiente da Política')
axes[1, 1].set_ylabel('Policy Loss')
axes[1, 1].set_xlabel('Iterações')

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('training_losses_comparison.png', dpi=300)

# =============================================================================
# Gráfico 2: Métricas de Estabilidade e Desempenho
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Métricas de Estabilidade e Desempenho', fontsize=20)

# Explained Variance
sns.lineplot(
    x='time/iterations',
    y='train/explained_variance',
    hue='Agent',
    data=train_df,
    ax=axes[0, 0],
    errorbar=None,
    linewidth=1.5
)
axes[0, 0].set_title('Variância Explicada')
axes[0, 0].set_ylabel('Explained Variance')
axes[0, 0].set_ylim(-0.1, 1.1)
axes[0, 0].set_xlabel('Iterações')

# Approx KL
sns.lineplot(
    x='time/iterations',
    y='train/approx_kl',
    hue='Agent',
    data=train_df,
    ax=axes[0, 1],
    errorbar=None,
    linewidth=1.5
)
axes[0, 1].set_title('Divergência KL Aproximada')
axes[0, 1].set_ylabel('KL Divergence')
axes[0, 1].set_yscale('log')  # Escala log para melhor visualização
axes[0, 1].set_xlabel('Iterações')

# Clip Fraction
sns.lineplot(
    x='time/iterations',
    y='train/clip_fraction',
    hue='Agent',
    data=train_df,
    ax=axes[1, 0],
    errorbar=None,
    linewidth=1.5
)
axes[1, 0].set_title('Fração de Clipping')
axes[1, 0].set_ylabel('Clip Fraction')
axes[1, 0].set_ylim(0, 1)
axes[1, 0].set_xlabel('Iterações')

# Learning Rate
sns.lineplot(
    x='time/iterations',
    y='train/learning_rate',
    hue='Agent',
    data=train_df,
    ax=axes[1, 1],
    errorbar=None,
    linewidth=1.5
)
axes[1, 1].set_title('Taxa de Aprendizado')
axes[1, 1].set_ylabel('Learning Rate')
axes[1, 1].set_xlabel('Iterações')

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
    plt.title('Correlação entre Value Loss e Recompensa Média (Últimas 50 iterações)', pad=20)
    plt.savefig('value_loss_reward_correlation.png', dpi=300)

plt.show()

print("Análise concluída! Gráficos salvos no diretório atual.")