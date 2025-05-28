import gymnasium as gym
from ultralytics import YOLO
import tempfile
import yaml
import numpy as np
import shutil
from pathlib import Path
import torch
from typing import List, Dict, Tuple

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
class YOLORLEnv(gym.Env):
    """Ambiente RL para Active Learning usando Entropia e Confiança do YOLOv11."""
    
    def __init__(self, yolo_model: YOLO, unlabeled_images: List[str], oracle_labels: Dict[str, list], batch_size: int = 32):
        super().__init__()
        
        self.yolo = yolo_model
        self.unlabeled_images = unlabeled_images.copy()
        self.oracle = oracle_labels
        self.batch_size = batch_size
        
        # Configuração do espaço de observação e ação
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32  # [entropia, confiança, progresso]
        )
        self.action_space = gym.spaces.Discrete(2)  # 0 = não rotular, 1 = rotular
        
        # Estado interno
        self.current_image_idx = 0
        self.selected_images = []
        self.steps_since_last_train = 0
        self.current_mAP = 0.3
        self.best_mAP = 0.3


    def reset(self, **kwargs):
        """Reinicia para um novo ciclo de seleção"""
        self.current_image_idx = np.random.randint(0, len(self.unlabeled_images))
        self.current_image_path = self.unlabeled_images[self.current_image_idx]
        
        entropy, confidence = self._extract_metrics(self.current_image_path)

        progress = len(self.selected_images) / self.batch_size
        
        return np.array([entropy, confidence, progress], dtype=np.float32)

    def step(self, action: int):
        reward = 0.0
        done = False
        info = {}
        
        if action == 1:  # Selecionar imagem
            if self.current_image_path not in self.selected_images:
                
                self.selected_images.append(self.current_image_path)
                # Recompensa imediata baseada na qualidade da seleção
                reward += self._calculate_immediate_reward()
                self._add_to_training_data(self.current_image_path)
                self.unlabeled_images.remove(self.current_image_path)
        
        # Atualizar contadores
        self.steps_since_last_train += 1
        limiteDeEntropias = len(self.unlabeled_images) / 3
        # Verificar se deve treinar
        if len(self.selected_images) >= int(limiteDeEntropias):
            old_mAP = self.current_mAP
            self._retrain_yolo()
            new_mAP = self._evaluate_model()
            
            # Recompensa principal baseada na melhoria do mAP
            reward += new_mAP - old_mAP
            self.current_mAP = new_mAP
            self.best_mAP = max(self.best_mAP, new_mAP)
            
            # Resetar para novo ciclo
            self.selected_images = []
            done = True
        
        # Próximo estado
        next_state = self.reset()
        
        return next_state, reward, done, False, info
    
    def _calculate_immediate_reward(self):
        """Recompensa baseada nas entropia e confiança."""
        entropy, confidence = self._extract_metrics(self.current_image_path)
        return (entropy * 0.7) + (confidence * 0.3)
    
    def _retrain_yolo(self):

        # Criar arquivo temporário com caminho absoluto
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "dataset_config.yaml"
            
            try:
                # Configurar dados
                data_config = {
                    'path': "coco",
                    'train': 'E:/COCO-Dataset/train2017/train/images/',
                    'val': 'E:/COCO-Dataset/val2017/images/',
                    'names': self.yolo.names,
                    'nc': self.yolo.model.nc
                }
                
                # Escrever arquivo YAML
                with open(temp_file, 'w') as f:
                    yaml.dump(data_config, f)
                
                # Verificar se arquivo foi criado
                if not temp_file.exists():
                    raise FileNotFoundError(f"Arquivo {temp_file} não foi gerado")

                # Treinar com caminho absoluto
                self.yolo.train(
                    data=str(temp_file),
                    epochs=self.total_epochs,
                    imgsz=480,
                    batch=8,
                    cache=False,
                    workers= 8,
                    device = 0,
                    dnn = True
                )
               
                # Atualizar mAP
                self.current_mAP = self.yolo.val(
                    data=str(temp_file),
                    batch=8,
                    device=DEVICE
                ).box.map50
                
            except Exception as e:
                print(f"Erro no re-treino: {str(e)}")
                if temp_file.exists():
                    print(f"Conteúdo do arquivo temporário:\n{temp_file.read_text()}")
        


    def _extract_metrics(self, image_path: str) -> Tuple[float, float]:
        """Extrai entropia e confiança média da imagem."""
        results = self.yolo.predict(image_path, verbose=False)
        
        if len(results) == 0:
            return 0.0, 0.0  # Caso sem detecções
        
        confs = results[0].boxes.conf.cpu().numpy()
        confs_normalized = confs / (confs.sum() + 1e-10)
        print(f"Confs: {confs_normalized}")

        # Caso 1: Sem detecções ou resultados inválidos
        if len(results[0].boxes) <= 0 or results[0].boxes.conf is None:
            return 0.0, 0.0  # Valores padrão
        else: 
            try:
                # Calcular entropia
                entropy = -np.sum(confs_normalized * np.log(confs_normalized + 1e-10))

                print(f"ENTROPIA {entropy}")
                
                # Calcular confiança média (média das confianças das detecções)
                avg_confidence = np.clip(confs.mean(), 0.0, 1.0)
                
                return entropy, avg_confidence
            except Exception as e:
                print(f"Erro ao processar {image_path}: {e}")
                return 0.0, 0.0
        

    def _add_to_training_data(self, image_path: str):
        # Caminhos de origem
        src_image_path = Path(image_path)
        src_label_path = Path(str(src_image_path).replace("images", "labels").replace(".jpg", ".txt"))
        
        # Caminhos de destino
        dest_image_dir = Path("E:/COCO-Dataset/train2017/train/images/")
        dest_label_dir = Path("E:/COCO-Dataset/train2017/train/labels/")
        
        # Garantir diretórios de destino
        dest_image_dir.mkdir(parents=True, exist_ok=True)
        dest_label_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Mover imagem
            shutil.move(str(src_image_path), str(dest_image_dir / src_image_path.name))
            # Mover label correspondente
            shutil.move(str(src_label_path), str(dest_label_dir / src_label_path.name))
            print(f"Movidos: {src_image_path.name} e label para treinamento")

        except Exception as e:
            print(f"Erro ao mover {src_image_path.name}: {str(e)}")
            # Reverter ambos se houver erro
            if (dest_image_dir / src_image_path.name).exists():
                shutil.move(str(dest_image_dir / src_image_path.name), str(src_image_path))
            if (dest_label_dir / src_label_path.name).exists():
                shutil.move(str(dest_label_dir / src_label_path.name), str(src_label_path))

    def _evaluate_model(self) -> float:
        # Avaliar em um dataset de validação
        results = self.yolo.val()
        return results.box.map50