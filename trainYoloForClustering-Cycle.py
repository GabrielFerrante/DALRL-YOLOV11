from ultralytics import YOLO
import os
import shutil
import torch
from tqdm import tqdm  # Para barra de progresso (opcional)

def move_images_and_labels(source_img_dir, source_label_dir, dest_img_dir, dest_label_dir, img_extensions=['.jpg', '.png', '.jpeg'], label_extension='.txt'):
    """
    Move imagens e seus arquivos de labels correspondentes para novos diretórios
    
    Args:
        source_img_dir: Diretório de origem das imagens
        source_label_dir: Diretório de origem dos labels
        dest_img_dir: Diretório de destino das imagens
        dest_label_dir: Diretório de destino dos labels
        img_extensions: Extensões de arquivo de imagem a serem consideradas
        label_extension: Extensão dos arquivos de label
    """
    
    # Criar diretórios de destino se não existirem
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_label_dir, exist_ok=True)
    
    # Contadores
    moved_images = 0
    moved_labels = 0
    
    # Listar todas as imagens no diretório de origem
    image_files = [
        f for f in os.listdir(source_img_dir) 
        if any(f.lower().endswith(ext) for ext in img_extensions)
    ]
    
    print(f"Encontradas {len(image_files)} imagens em {source_img_dir}")
    
    # Processar cada imagem
    for img_file in tqdm(image_files, desc="Movendo arquivos"):
        # Construir caminhos completos
        img_path = os.path.join(source_img_dir, img_file)
        img_name = os.path.splitext(img_file)[0]
        label_file = img_name + label_extension
        label_path = os.path.join(source_label_dir, label_file)
        
        # Verificar se o arquivo de label existe
        if not os.path.exists(label_path):
            print(f"⚠️ Aviso: Label não encontrado para {img_file}")
            continue
        
        # Destinos
        dest_img_path = os.path.join(dest_img_dir, img_file)
        dest_label_path = os.path.join(dest_label_dir, label_file)
        
        try:
            # Mover imagem
            shutil.move(img_path, dest_img_path)
            moved_images += 1
            
            # Mover label
            shutil.move(label_path, dest_label_path)
            moved_labels += 1
        except Exception as e:
            print(f"❌ Erro movendo {img_file}: {str(e)}")
    
    # Relatório final
    print(f"\n✅ Operação concluída:")
    print(f"- Imagens movidas: {moved_images}/{len(image_files)}")
    print(f"- Labels movidos: {moved_labels}")
    print(f"- Destino imagens: {dest_img_dir}")
    print(f"- Destino labels: {dest_label_dir}")




if __name__ == '__main__':

    # Limpeza do cache da GPU (se disponível)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats() 

    # Configuração do modelo (supondo que 'yolov11x' é a versão completa)
    model = YOLO(model="yolo11x.yaml")  #Sem pre-treinamento

    # Configurar caminhos (ajuste conforme necessário)
    SOURCE_IMAGE_DIR = "./selected_images_clustering/images/"
    SOURCE_LABEL_DIR = "./selected_images_clustering/labels/"
    DEST_IMAGE_DIR = "F:/COCO-Dataset/train2017/clustering/train/images/"
    DEST_LABEL_DIR = "F:/COCO-Dataset/train2017/clustering/train/labels/"
    
    # Executar a função
    move_images_and_labels(
        source_img_dir=SOURCE_IMAGE_DIR,
        source_label_dir=SOURCE_LABEL_DIR,
        dest_img_dir=DEST_IMAGE_DIR,
        dest_label_dir=DEST_LABEL_DIR,
        img_extensions=['.jpg', '.jpeg', '.png', '.bmp'],
        label_extension='.txt'
    )


    # Treinar
    results = model.train(
        data="cocoClustering.yaml",
        epochs=50,
        imgsz=480,
        batch=8,
        lr0= 0.001,
        optimizer ='AdamW',
        cache=False,
        cos_lr = True,
        pretrained=False,
        workers= 8,
        device = 0,
        dnn = True,
        project = "Yolov11-WithClustersSamples",
        plots = True
        
    )