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



# Configurações
CUDA_DEVICE = "cuda:0"  # Dispositivo CUDA
PPO_MODEL_PATH = "./logs-clustering/active_learning_20250609_192355/best_model-ClusteringPool/best_model"  # Caminho para o agente PPO treinado
YOLO_MODEL_PATH = "runs/detect/yolov11-initial-WithClusteringSamples/weights/best.pt"  # Caminho para o modelo YOLO
POOL_DIR = "F:/COCO-Dataset/train2017/clustering/pool/images/"  # Diretório com novas imagens não rotuladas
LABEL_DIR = "F:/COCO-Dataset/train2017/clustering/pool/labels/"
BUDGET = 945  # Orçamento de seleção, corresponde 10% do pool (94630 / 100 * 10)
OUTPUT_DIR = "selected_images_clustering/images/"  # Diretório para salvar imagens selecionadas
OUTPUT_LABEL_DIR = "selected_images_clustering/labels/"
LOG_FILE = "selection_clusterSamples_log.json"  # Arquivo para registrar seleções

def process_image(yolo_model: YOLO, img_path: str) -> Tuple[float, float]:
    """Processa uma imagem com YOLO e retorna entropia e confiança média"""
    img = cv2.imread(img_path)
    
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
            if 0 in confs:
                print(f"Erro ao processar imagem: {img_path}")
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

def select_images(ppo_agent, yolo_model, image_paths: List[str], budget: int) -> Tuple[List[str], List[Dict]]:
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
    """Copia as imagens selecionadas para o diretório de saída"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
    
    for img_path in selected_paths:
        img = cv2.imread(img_path)
        if img is not None:
            output_path = os.path.join(output_dir, os.path.basename(img_path))
            cv2.imwrite(output_path, img)
            shutil.copyfile(label_dir + os.path.basename(img_path).split('.')[0] + ".txt", OUTPUT_LABEL_DIR + os.path.basename(img_path).split('.')[0] + ".txt")
            print(f"💾 Salvo: {output_path}")

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
    
    # Carregar caminhos das imagens
    image_files = sorted([
        os.path.join(POOL_DIR, f) 
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
    
    # Salvar log de seleção
    with open(LOG_FILE, "w") as f:
        json.dump(selection_log, f, indent=2)
    
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
    #Diversidade = Número de índices únicos selecionados / Total de imagens selecionadas
    # Quanto menor o valor, maior a repetição de seleções
    #print(f"- Diversidade: {len(selected_entropies)/len(selected_paths):.2%} de imagens únicas")

if __name__ == "__main__":
    # Registrar tempo de execução
    start_time = datetime.now()
    print(f"⏰ Início do processo: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    main()
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"⏱️ Tempo total de execução: {duration.total_seconds():.2f} segundos")
    print(f"🏁 Fim do processo: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")