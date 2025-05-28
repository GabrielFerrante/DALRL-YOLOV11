from typing import Optional, Union
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
import gymnasium as gym
class ActiveLearningEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env: gym.Env,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_model_save_path = best_model_save_path
        self.deterministic = deterministic
        self.best_mean_reward = -np.inf
        
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            mean_reward = self._evaluate_model()
            
            # Salvar melhor modelo
            if mean_reward > self.best_mean_reward and self.best_model_save_path is not None:
                self.best_mean_reward = mean_reward
                self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                if self.verbose >= 1:
                    print(f"Novo melhor modelo salvo com recompensa média: {self.best_mean_reward:.2f}")

            # Log de progresso
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/best_mean_reward", self.best_mean_reward)
            
        return True

    def _evaluate_model(self) -> float:
        rewards = []
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, done, _, _ = self.eval_env.step(action)
                episode_reward += reward
            
            rewards.append(episode_reward)
        
        mean_reward = np.mean(rewards)
        if self.verbose >= 1:
            print(f"Recompensa média de avaliação após {self.n_calls} steps: {mean_reward:.2f}")
        
        return mean_reward
    
class TrainingCallback(BaseCallback):
    def __init__(self, save_freq: int = 100, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        
    def _on_step(self) -> bool:
        # Salvar PPO periodicamente
        if self.n_calls % self.save_freq == 0:
            self.model.save(f"ppo_agent_checkpoint_{self.n_calls}")
            
        # Salvar YOLO (opcional)
        if self.n_calls % 500 == 0:
            self.training_env.envs[0].yolo.save(f"yolo_after_{self.n_calls}_steps.pt")
            
        return True