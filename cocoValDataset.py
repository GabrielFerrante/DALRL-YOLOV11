from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import cv2
from pathlib import Path
from tqdm.notebook import tqdm
class CocoValDataset(Dataset):
    def __init__(self, base_path, img_size=480):
        self.img_dir = Path(base_path) / 'val' / 'images'
        self.img_files = list(self.img_dir.glob('*.jpg'))
        
        # Transformações ajustadas
        self.transform = transforms.Compose([
            transforms.ToTensor(),         # Converte para tensor [0.0, 1.0] e divide por 255
            transforms.Resize((img_size, img_size)),
            
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        
        # Carregar com OpenCV e converter para RGB
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Converter para PIL Image para garantir compatibilidade
        from PIL import Image
        img_pil = Image.fromarray(img)
        
        # Aplicar transformações
        tensor_img = self.transform(img_pil)  # Agora em [0.0, 1.0]
        
        return tensor_img, img_path.name

    def calculate_statistics(self):
        """Calcula média e desvio padrão do dataset inteiro"""
        # Inicializar acumuladores
        mean_sum = torch.zeros(3)
        std_sum = torch.zeros(3)
        total_pixels = 0
        
        # Iterar por todas as imagens
        for img_path in tqdm(self.img_files, desc='Calculando estatísticas'):
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor_img = self.base_transform(img)  # Tensor [3, H, W]
            
            # Acumular estatísticas
            mean_sum += tensor_img.sum(dim=[1, 2])  # Soma por canal
            std_sum += (tensor_img ** 2).sum(dim=[1, 2])  # Soma quadrados
            total_pixels += tensor_img.shape[1] * tensor_img.shape[2]  # H * W
        
        # Calcular valores finais
        mean = mean_sum / total_pixels
        std = torch.sqrt((std_sum / total_pixels) - (mean ** 2))
        
        return mean.tolist(), std.tolist()
