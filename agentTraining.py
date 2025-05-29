import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from typing import Tuple, List, Dict, Any
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from ultralytics import YOLO 
import cv2
import os

BUDGET = 100              # 10% do pool de 1000 imagens
NUM_ENVS = 4              # Número de ambientes paralelos
TIMESTEPS = 100000        # Total de passos de treino
POOL_DIR = "E:/COCO-Dataset/train2017/val/images/"  # Diretório com imagens

class ActiveLearningEnv(gym.Env):
    def __init__(self, yolo_model: YOLO, image_paths: List[str], budget: int):
        super().__init__()
        
        self.yolo_model = yolo_model
        self.image_paths = image_paths
        self.budget = budget
        self.num_images = len(image_paths)
        
        # Espaço de observação: [entropia, confiança_média, orçamento_restante]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([1, 1, budget]),
            dtype=np.float32
        )
        
        # Espaço de ação: 2 ações (0 = não selecionar, 1 = selecionar)
        self.action_space = spaces.Discrete(2)
        
        self.reset()

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        # Embaralha as imagens a cada novo episódio
        self.indices = np.random.permutation(self.num_images)
        self.current_idx = 0
        self.remaining_budget = self.budget
        self.selected_indices = []
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        terminated = truncated = False
        reward = 0.0
        
        # Processar imagem atual com YOLO
        entropy, avg_confidence = self._process_image(self.indices[self.current_idx])
        
        # Ação de seleção (1) com orçamento disponível
        if action == 1 and self.remaining_budget > 0:
            self.selected_indices.append(self.indices[self.current_idx])
            self.remaining_budget -= 1
            # Recompensa: combinação de entropia e confiança
            reward = entropy * (1 - avg_confidence)  # Maximiza entropia e minimiza confiança
            
        # Penalização por seleção sem orçamento
        elif action == 1 and self.remaining_budget <= 0:
            reward = -0.5
        
        # Avança para próxima imagem
        self.current_idx += 1
        
        # Verifica término do episódio
        if self.current_idx >= self.num_images or self.remaining_budget <= 0:
            terminated = True
            # Recompensa final baseada na diversidade de seleção
            if len(self.selected_indices) > 0:
                reward += 0.1 * len(set(self.selected_indices)) / len(self.selected_indices)
        
        # Prepara próxima observação
        next_obs = self._get_obs()
        return next_obs, reward, terminated, truncated, {}

    def _get_obs(self) -> np.ndarray:
        """Retorna a observação atual processando a imagem"""
        entropy, avg_confidence = self._process_image(self.indices[self.current_idx])
        return np.array([entropy, avg_confidence, self.remaining_budget], dtype=np.float32)

    def _process_image(self, idx: int) -> Tuple[float, float]:
        """Processa uma imagem com YOLO e retorna entropia e confiança média"""
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Erro ao carregar imagem: {img_path}")
            return (0.0, 0.0)
        
        # Executar predição com YOLO
        results = self.yolo_model(img, verbose=False)
        
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


def main():
    # Configurações
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    yolo = YOLO("runs/detect/yolov11-initial/weights/best.pt").to(DEVICE)

    # Carregar caminhos das imagens
    image_files = sorted([
        os.path.join("E:/COCO-Dataset/train2017/val/images/", f) 
        for f in os.listdir("E:/COCO-Dataset/train2017/val/images/") 
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])
    print(f"Encontradas {len(image_files)} imagens no pool")

    """
    images_in_pool = [f"E:/COCO-Dataset/train2017/val/images/{f}" for f in os.listdir("E:/COCO-Dataset/train2017/val/images/")]
    oracle_labels = {img: img.replace(".jpg",".txt").replace("images","labels") for img in images_in_pool}  # Preencher com anotações reais
    """

    from agentTraining import ActiveLearningEnv
    # Criar ambiente
    env = make_vec_env(
            lambda: ActiveLearningEnv(yolo, image_files, BUDGET),
            n_envs=NUM_ENVS,
            vec_env_cls=DummyVecEnv
    )
        # Configurar política do PPO
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]
    )
        
        # Inicializar agente PPO
    agent = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=DEVICE,
            learning_rate=3e-4,
            n_steps=2048 // NUM_ENVS,
            batch_size=8,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02
    )
        
    # Treinar o agente
    agent.learn(total_timesteps=TIMESTEPS, progress_bar=True)
        
    # Salvar modelo treinado
    agent.save("active_learning_agent")
    print("Agente treinado e salvo com sucesso!")


if __name__ == "__main__":
    main()