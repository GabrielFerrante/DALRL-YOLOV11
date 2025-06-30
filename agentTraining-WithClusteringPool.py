import os
import cv2
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecMonitor
from typing import Tuple, List, Dict, Any
from ultralytics import YOLO
import time
from datetime import datetime

BUDGET = 945  # Orçamento de seleção, corresponde 1% do pool (94630 / 100 * 1)
NUM_ENVS = 4              # Número de ambientes paralelos
TIMESTEPS = 100352        # Total de passos de treino (multiplo de 512)
POOL_DIR = "F:/COCO-Dataset/train2017/clustering/pool/images/"  # Diretório com imagens
LOG_DIR = "logs-clustering" 

class TensorBoardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "mean_entropy" in info:
                self.logger.record("custom/mean_entropy", info["mean_entropy"])
            if "mean_confidence" in info:
                self.logger.record("custom/mean_confidence", info["mean_confidence"])
            if "images_selected" in info:
                self.logger.record("custom/images_selected", info["images_selected"])
        return True

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

        # Variáveis para logging
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_selected = []
        self.episode_entropy = []
        self.episode_confidence = []
        
        self.reset()

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        # Embaralha as imagens a cada novo episódio
        self.indices = np.random.permutation(self.num_images)
        self.current_idx = 0
        self.remaining_budget = self.budget
        self.selected_indices = []
        self.current_entropy = 0.0
        self.current_confidence = 0.0
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
             # Registrar métricas do episódio
            self.episode_rewards.append(self.cumulative_reward)
            self.episode_lengths.append(self.current_idx)
            self.episode_selected.append(len(self.selected_indices))
            self.episode_entropy.append(self.cumulative_entropy / self.current_idx)
            self.episode_confidence.append(self.cumulative_confidence / self.current_idx)

        # Atualizar métricas cumulativas
        self.cumulative_reward = reward if not hasattr(self, 'cumulative_reward') else self.cumulative_reward + reward
        self.cumulative_entropy = entropy if not hasattr(self, 'cumulative_entropy') else self.cumulative_entropy + entropy
        self.cumulative_confidence = avg_confidence if not hasattr(self, 'cumulative_confidence') else self.cumulative_confidence + avg_confidence

        # Prepara próxima observação
        next_obs = self._get_obs()
        return next_obs, reward, terminated, truncated, {
                                                            "episode": {
                                                                "r": self.cumulative_reward,
                                                                "l": self.current_idx
                                                            },
                                                            "mean_entropy": self.cumulative_entropy / self.current_idx,
                                                            "mean_confidence": self.cumulative_confidence / self.current_idx,
                                                            "images_selected": len(self.selected_indices)
                                                        }

    def _get_obs(self) -> np.ndarray:
        """Retorna a observação atual processando a imagem"""
        entropy, avg_confidence = self._process_image(self.indices[self.current_idx])
        return np.array([entropy, avg_confidence, self.remaining_budget], dtype=np.float32)

    def _process_image(self, idx: int) -> Tuple[float, float]:
        """Processa uma imagem com YOLO e retorna entropia e confiança média"""
        img_path = self.image_paths[idx]

        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Erro ao carregar imagem: {img_path}")
                return (0.0, 0.0)
            
            # Restante do processamento...
            
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
            
        except Exception as e:
            print(f"Erro crítico ao processar {img_path}: {str(e)}")
            return (0.0, 0.0)


def main():

    # Limpeza do cache da GPU (se disponível)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("MEMÓRIA LIMPA")
    
    # Configurações

    caminho = "./Yolov11-WithClustersSamples/"  # Diretório atual (substitua pelo caminho desejado)
    itens = os.listdir(caminho)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    yolo = YOLO(f"Yolov11-WithClustersSamples/{itens[len(itens)-1]}/weights/best.pt").to(DEVICE)

    def make_env_fn( image_paths, budget):
        def _init():
            return ActiveLearningEnv(yolo, image_paths, budget)
        return _init

    # Criar diretório de logs com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"active_learning_{timestamp}")
    os.makedirs(log_path, exist_ok=True)

    # Configurar logger
    from stable_baselines3.common.logger import configure
    logger = configure(log_path, ["stdout", "csv", "tensorboard"])

    # Carregar caminhos das imagens
    image_files = sorted([
        os.path.join("F:/COCO-Dataset/train2017/clustering/pool/images/", f) 
        for f in os.listdir("F:/COCO-Dataset/train2017/clustering/pool/images/") 
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])
    print(f"Encontradas {len(image_files)} imagens no pool")


    env_fns = [make_env_fn(image_files, BUDGET) for _ in range(NUM_ENVS)]
    env = SubprocVecEnv(env_fns = env_fns)
    env = VecMonitor(env)
    
    # Configurar política do PPO
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=(dict(pi=[256, 256], vf=[256, 256]))
    )
        
        # Inicializar agente PPO
    agent = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=DEVICE,
            learning_rate=1e-4,
            n_steps= 512,
            batch_size=16,
            max_grad_norm=0.5,
            seed=42,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            tensorboard_log=log_path
    )
    
    # Configurar logger personalizado
    agent.set_logger(logger)

    # Callbacks
    eval_callback = EvalCallback(
        env,
        best_model_save_path=os.path.join(log_path, "best_model-ClusteringPool"),
        log_path=log_path,
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    tb_callback = TensorBoardCallback()

    start_time = time.time()
    # Treinar o agente
    agent.learn(
        total_timesteps=TIMESTEPS, 
        progress_bar=True,
        callback=[eval_callback, tb_callback],
        tb_log_name="active_learning"
        )

    training_time = time.time() - start_time
        
    # Salvar modelo treinado
    agent.save(os.path.join(log_path, "active_learning_agent"))
    print(f"Agente treinado e salvo em: {log_path}")
    print(f"Tempo de treinamento: {training_time/60:.2f} minutos")
    

if __name__ == "__main__":
    main()