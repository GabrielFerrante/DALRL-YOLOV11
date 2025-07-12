import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from stable_baselines3 import PPO
from typing import List, Tuple, Dict
from datetime import datetime
import json
import shutil

# Configurar ambiente para UTF-8
os.environ["PYTHONUTF8"] = "1"

caminhoYolo = "./Yolov11-WithClustersSamples/"  # Diretório atual (substitua pelo caminho desejado)
itensYolo = os.listdir(caminhoYolo)

caminhoAgents = "./logs-clustering"  # Diretório atual (substitua pelo caminho desejado)
itensAgents = os.listdir(caminhoAgents)

# Configurações
CUDA_DEVICE = "cuda:0"  # Dispositivo CUDA
#PPO_MODEL_PATH = "./logs-clustering/active_learning_20250609_192355/best_model-ClusteringPool/best_model"  # Caminho para o agente PPO treinado
PPO_MODEL_PATH = os.path.join("./logs-clustering", itensAgents[-1], "best_model-ClusteringPool", "best_model")
#YOLO_MODEL_PATH = "runs/detect/yolov11-initial-WithClusteringSamples/weights/best.pt"  # Caminho para o modelo YOLO
YOLO_MODEL_PATH = os.path.join("Yolov11-WithClustersSamples", itensYolo[-1], "weights", "best.pt")
POOL_DIR = "F:/COCO-Dataset/train2017/clustering/pool/images/"  # Diretório com novas imagens não rotuladas
LABEL_DIR = "F:/COCO-Dataset/train2017/clustering/pool/labels/"
BUDGET = 945  # Orçamento de seleção, corresponde 10% do pool (94583 / 100 * 10)
OUTPUT_DIR = "selected_images_clustering/images/"  # Diretório para salvar imagens selecionadas
OUTPUT_LABEL_DIR = "selected_images_clustering/labels/"
LOG_FILE = "selection_clusterSamples_log.json"  # Arquivo para registrar seleções



def process_image(yolo_model: YOLO, img_path: str) -> Tuple[float, float]:
    """
    Processes an image using a YOLO model and returns the average entropy and average confidence of detected bounding boxes.
    Args:
        yolo_model (YOLO): The YOLO model instance used for object detection.
        img_path (str): The file path to the image to be processed.
    Returns:
        Tuple[float, float]: A tuple containing:
            - avg_entropy (float): The average entropy calculated from the confidence scores of detected bounding boxes.
            - avg_confidence (float): The average confidence score of detected bounding boxes.
    Notes:
        - If the image cannot be loaded or no detections are found, both values returned will be 0.0.
        - Entropy is calculated for each confidence score using the binary entropy formula.
        - Handles exceptions and prints error messages for critical failures.
    """
    """Processa uma imagem com YOLO e retorna entropia e confiança média"""
    try:
        # Usar imdecode para lidar melhor com caminhos especiais
        img_data = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"Erro ao carregar imagem: {img_path}")
            return (0.0, 0.0)
        
        # Executar predição com YOLO
        results = yolo_model(img, verbose=False)
        
        confidences = []
        entropies = []
        
        for result in results:
            if result.boxes is not None:
                # Coletar confianças das bounding boxes
                confs = result.boxes.conf.cpu().numpy()
                if len(confs) == 0:
                    print(f"Sem detecções em: {img_path}")
                    return (0.0, 0.0)
                else:
                    confidences.extend(confs)
                    for conf in confs:
                        # Calcular entropia para cada confiança
                        entropy = -conf * np.log2(conf + 1e-10) - (1-conf) * np.log2(1-conf + 1e-10)
                        entropies.append(entropy)
            
        # Calcular entropia média
        avg_entropy = float(np.mean(entropies)) if entropies else 0.0
            
        # Calcular confiança média
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        
        return (avg_entropy, avg_confidence)
        
    except Exception as e:
        print(f"Erro crítico ao processar {img_path}: {str(e)}")
        return (0.0, 0.0)

def select_images(ppo_agent, yolo_model, image_paths: List[str], budget: int) -> Tuple[List[str], List[Dict]]:
    """
    Selects images using a trained PPO agent and a YOLO model within a specified budget.
    For each image in the provided list, the function processes the image with the YOLO model to obtain entropy and average confidence.
    These metrics, along with the remaining budget, are used as observations for the PPO agent, which decides whether to select the image.
    The process continues until the budget is exhausted or all images have been considered.
    Args:
        ppo_agent: The trained PPO agent used to make selection decisions.
        yolo_model: The YOLO model used to process images and extract features.
        image_paths (List[str]): List of file paths to the images to be considered for selection.
        budget (int): The maximum number of images that can be selected.
    Returns:
        Tuple[List[str], List[Dict]]: 
            - A list of selected image paths.
            - A log (list of dictionaries) detailing the selection process for each image, including entropy, average confidence, action taken, remaining budget, and selection status.
    """
    """Seleciona imagens usando o agente PPO treinado"""
    selected_paths = []
    selection_log = []
    remaining_budget = budget
    
    for img_path in image_paths:
        # Processar imagem com YOLO
        entropy, avg_confidence = process_image(yolo_model, img_path)
        
        # Criar observação para o agente PPO
        obs = np.array([entropy, avg_confidence, remaining_budget], dtype=np.float32)
        
        # Prever ação do agente
        action, _ = ppo_agent.predict(obs, deterministic=True)
        
        # Registrar decisão
        log_entry = {
            "image_path": img_path,
            "entropy": float(entropy),
            "avg_confidence": float(avg_confidence),
            "action": int(action),
            "remaining_budget": int(remaining_budget),
            "selected": False
        }
        
        # Executar ação de seleção
        if action == 1 and remaining_budget > 0:
            selected_paths.append(img_path)
            remaining_budget -= 1
            log_entry["selected"] = True
            print(f"✅ Selecionada: {os.path.basename(img_path)} | Entropia: {entropy:.4f}, Conf: {avg_confidence:.4f}, Orçamento: {remaining_budget}/{budget}")
        
        selection_log.append(log_entry)
        
        # Parar se o orçamento se esgotar
        if remaining_budget <= 0:
            print("🚫 Orçamento esgotado!")
            break
    
    return selected_paths, selection_log

def save_selected_images(selected_paths: List[str], output_dir: str, label_dir: str):
    """
    Copies selected images and their corresponding label files to specified output directories.
    Args:
        selected_paths (List[str]): List of file paths to the selected images.
        output_dir (str): Directory where the selected images will be moved.
        label_dir (str): Directory containing the label files corresponding to the images.
    Behavior:
        - Moves each image in `selected_paths` to `output_dir`.
        - Moves the corresponding label file (with the same base name and `.txt` extension) from `label_dir` to `OUTPUT_LABEL_DIR`.
        - Creates `output_dir` and `OUTPUT_LABEL_DIR` if they do not exist.
        - Prints a warning if a label file is not found for an image.
        - Prints a confirmation message for each successfully moved image.
        - Prints an error message if an exception occurs during the move operation.
    """
    """Copia as imagens selecionadas para o diretório de saída"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
    
    for img_path in selected_paths:
        try:
            # Usar shutil.copy2 para preservar metadados
            dst_img = os.path.join(output_dir, os.path.basename(img_path))
            shutil.move(img_path, dst_img)
            
            # Construir caminho do label correspondente
            label_src = os.path.join(label_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
            dst_label = os.path.join(OUTPUT_LABEL_DIR, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
            
            if os.path.exists(label_src):
                shutil.move(label_src, dst_label)
            else:
                print(f"⚠️ Label não encontrado: {label_src}")
            
            print(f"💾 Salvo: {dst_img}")
        except Exception as e:
            print(f"Erro ao copiar {img_path}: {str(e)}")

def main():
    # Configurar dispositivo
    device = torch.device(CUDA_DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Carregar agente PPO treinado
    print("⏳ Carregando agente PPO...")
    ppo_agent = PPO.load(PPO_MODEL_PATH, device=device)
    print("✅ Agente PPO carregado com sucesso!")
    
    # Carregar modelo YOLO
    print("⏳ Carregando modelo YOLO...")
    yolo_model = YOLO(YOLO_MODEL_PATH).to(device)
    print("✅ Modelo YOLO carregado com sucesso!")
    
    # Carregar caminhos das imagens (usando caminhos normalizados)
    image_files = sorted([
        os.path.normpath(os.path.join(POOL_DIR, f)) 
        for f in os.listdir(POOL_DIR) 
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ])
    
    if not image_files:
        raise ValueError(f"❌ Nenhuma imagem encontrada em {POOL_DIR}")
    
    print(f"📂 Encontradas {len(image_files)} imagens no pool")
    
    # Selecionar imagens com o agente PPO
    print("\n🔍 Iniciando processo de seleção...")
    selected_paths, selection_log = select_images(ppo_agent, yolo_model, image_files, BUDGET)
    
    # Salvar imagens selecionadas
    save_selected_images(selected_paths, OUTPUT_DIR, LABEL_DIR)
    
    # Salvar log de seleção com UTF-8
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(selection_log, f, indent=2, ensure_ascii=False)
    
    # Gerar relatório
    print("\n📊 Relatório Final:")
    print(f"- Total de imagens no pool: {len(image_files)}")
    print(f"- Imagens selecionadas: {len(selected_paths)}")
    print(f"- Orçamento restante: {BUDGET - len(selected_paths)}")
    print(f"- Log de seleção salvo em: {LOG_FILE}")
    print(f"- Imagens selecionadas salvas em: {OUTPUT_DIR}")
    
    # Calcular estatísticas
    entropies = [entry["entropy"] for entry in selection_log]
    confidences = [entry["avg_confidence"] for entry in selection_log]
    selected_entropies = [entry["entropy"] for entry in selection_log if entry["selected"]]
    selected_confidences = [entry["avg_confidence"] for entry in selection_log if entry["selected"]]
    
    print("\n📈 Estatísticas das imagens selecionadas:")
    print(f"- Entropia média: {np.mean(selected_entropies):.4f} (Pool: {np.mean(entropies):.4f})")
    print(f"- Confiança média: {np.mean(selected_confidences):.4f} (Pool: {np.mean(confidences):.4f})")

if __name__ == "__main__":
    # Registrar tempo de execução
    start_time = datetime.now()
    print(f"⏰ Início do processo: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        main()
    except Exception as e:
        print(f"❌ ERRO GRAVE: {str(e)}")
        # Salvar traceback para análise
        import traceback
        with open("error_log.txt", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"⏱️ Tempo total de execução: {duration.total_seconds():.2f} segundos")
    print(f"🏁 Fim do processo: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")