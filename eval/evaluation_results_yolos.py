import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Configurações iniciais
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'

# Lista de cores para diferenciar os modelos
colors = plt.cm.tab20.colors
"""
arquivos = [
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

arquivos = [
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
"""
arquivos = [
    "../Yolov11-BASELINE-118k/train/results.csv"
]

# Verificar se encontrou arquivos
if not arquivos:
    raise FileNotFoundError("Nenhum arquivo CSV encontrado!")

# Dicionário para armazenar os DataFrames
dataframes = {}

# Ler cada arquivo CSV
for i, arquivo in enumerate(arquivos):
    nome_modelo = os.path.splitext(os.path.basename(arquivo))[0]
    try:
        df = pd.read_csv(arquivo)
        # Verificar colunas essenciais
        colunas_necessarias = ['epoch', 'metrics/precision(B)', 'metrics/recall(B)', 
                              'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
                              'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
                              'val/box_loss', 'val/cls_loss', 'val/dfl_loss']
        if not all(col in df.columns for col in colunas_necessarias):
            missing = [col for col in colunas_necessarias if col not in df.columns]
            print(f"Aviso: Arquivo {arquivo} está faltando colunas: {missing}")
            continue
        dataframes[f"cycle_{i}"] = df
    except Exception as e:
        print(f"Erro ao ler {arquivo}: {str(e)}")

# Função para plotar métricas
def plotar_metricas(metricas, titulo, ylabel):
    plt.figure()
    for i, (modelo, df) in enumerate(dataframes.items()):
        plt.plot(df['epoch'], df[metricas], 
                 label=modelo, 
                 color=colors[i % len(colors)],
                 linewidth=2)
    plt.title(titulo)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{titulo.replace(' ', '_')}.png", dpi=300)
    plt.show()

# Plotar métricas de avaliação
plotar_metricas('metrics/precision(B)', 'Precision (Bounding Box) per epoch', 'Precision')
plotar_metricas('metrics/recall(B)', 'Recall (Bounding Box) per epoch', 'Recall')
plotar_metricas('metrics/mAP50(B)', 'mAP50 (Bounding Box) per epoch', 'mAP@0.5')
plotar_metricas('metrics/mAP50-95(B)', 'mAP50-95 (Bounding Box) per epoch', 'mAP@0.5:0.95')

# Plotar losses de treino e validação
def plotar_losses():
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    tipos_loss = ['box_loss', 'cls_loss', 'dfl_loss']
    
    for j, loss_type in enumerate(tipos_loss):
        # Treino
        ax = axs[j, 0]
        for i, (modelo, df) in enumerate(dataframes.items()):
            ax.plot(df['epoch'], df[f'train/{loss_type}'], 
                    color=colors[i % len(colors)],
                    label=modelo)
        ax.set_title(f'Train {loss_type}')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        
        # Validação
        ax = axs[j, 1]
        for i, (modelo, df) in enumerate(dataframes.items()):
            ax.plot(df['epoch'], df[f'val/{loss_type}'], 
                    color=colors[i % len(colors)],
                    label=modelo)
        ax.set_title(f'Validation {loss_type}')
        ax.grid(True, alpha=0.3)
    
    for ax in axs.flat:
        ax.set_xlabel('Epoch')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=8)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig("Losses_training_yolos.png", dpi=300)
    plt.show()

plotar_losses()
