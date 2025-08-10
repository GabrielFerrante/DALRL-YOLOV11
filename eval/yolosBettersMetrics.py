import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Configurações iniciais
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans'})


"""
csv_files = [
    "../Yolov11-WithRandomSamples/train/results.csv",
    "../Yolov11-WithRandomSamples/train2/results.csv",
    "../Yolov11-WithRandomSamples/train3/results.csv",
    "../Yolov11-WithRandomSamples/train4/results.csv",
    "../Yolov11-WithRandomSamples/train5/results.csv",
    "../Yolov11-WithRandomSamples/train6/results.csv",
    "../Yolov11-WithRandomSamples/train7/results.csv",
    "../Yolov11-WithRandomSamples/train8/results.csv",
    "../Yolov11-WithRandomSamples/train9/results.csv",
    "../Yolov11-WithRandomSamples/train10/results.csv",
    "../Yolov11-WithRandomSamples/train11/results.csv",

]
"""
csv_files = [
    "../Yolov11-WithClustersSamples/train/results.csv",
    "../Yolov11-WithClustersSamples/train2/results.csv",
    "../Yolov11-WithClustersSamples/train3/results.csv",
    "../Yolov11-WithClustersSamples/train4/results.csv",
    "../Yolov11-WithClustersSamples/train5/results.csv",
    "../Yolov11-WithClustersSamples/train6/results.csv",
    "../Yolov11-WithClustersSamples/train7/results.csv",
    "../Yolov11-WithClustersSamples/train8/results.csv",
    "../Yolov11-WithClustersSamples/train9/results.csv",
    "../Yolov11-WithClustersSamples/train10/results.csv",
    "../Yolov11-WithClustersSamples/train11/results.csv",
    
    
]



if not csv_files:
    raise FileNotFoundError("Nenhum arquivo CSV encontrado no diretório!")

# Dicionário para armazenar os resultados
results = {
    'Modelo': [],
    'Precisão': [],
    'Recall': [],
    'mAP50': [],
    'mAP50-95': []
}

i = 0
# Processa cada arquivo CSV
for file in csv_files:
    df = pd.read_csv(file)
    
    # Remove espaços extras nos nomes das colunas
    df.columns = df.columns.str.strip()
    
    # Verifica se as colunas necessárias existem
    required_columns = [
        'metrics/precision(B)',
        'metrics/recall(B)',
        'metrics/mAP50(B)',
        'metrics/mAP50-95(B)'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Coluna '{col}' não encontrada no arquivo {file}")
    
    # Obtém o nome do modelo sem a extensão
    model_name = os.path.splitext(os.path.basename(file))[0]

    if i == 0:
        model_name = "base"
    else:
        model_name = f"cycle_{i}"
    
    # Armazena os valores máximos
    results['Modelo'].append(model_name)
    results['Precisão'].append(df['metrics/precision(B)'].max())
    results['Recall'].append(df['metrics/recall(B)'].max())
    results['mAP50'].append(df['metrics/mAP50(B)'].max())
    results['mAP50-95'].append(df['metrics/mAP50-95(B)'].max())

    i = i+ 1

# Cria DataFrame com os resultados
results_df = pd.DataFrame(results)

# Configuração do gráfico
fig, ax = plt.subplots(figsize=(14, 8), dpi=100)
bar_width = 0.2
index = range(len(results_df))

# Cria as barras para cada métrica
bar1 = ax.bar(index, results_df['Precisão'], bar_width, label='Precision', alpha=0.85)
bar2 = ax.bar([i + bar_width for i in index], results_df['Recall'], bar_width, label='Recall', alpha=0.85)
bar3 = ax.bar([i + bar_width*2 for i in index], results_df['mAP50'], bar_width, label='mAP50', alpha=0.85)
bar4 = ax.bar([i + bar_width*3 for i in index], results_df['mAP50-95'], bar_width, label='mAP50-95', alpha=0.85)

# Configurações do eixo X
ax.set_xlabel('Models')
ax.set_ylabel('Max values')
ax.set_title('Comparison of YOLOv11 Models Metrics')
ax.set_xticks([i + bar_width*1.5 for i in index])
ax.set_xticklabels(results_df['Modelo'], rotation=45, ha='right')
ax.legend(loc='best', framealpha=0.9)

# Adiciona valores nas barras
for bars in [bar1, bar2, bar3, bar4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

# Ajustes finais
plt.ylim(0, 1.05)  # Como as métricas são entre 0-1
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()