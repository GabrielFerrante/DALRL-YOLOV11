from ultralytics import YOLO
import os
import shutil
import torch
from tqdm import tqdm  # Para barra de progresso (opcional)

if __name__ == '__main__':

    #EXECUTAR ESTE ARQUIVO FORA DO CICLO. SOMENTE PARA FINALIZAR ALGUM TREINAMENTO

    # Limpeza do cache da GPU (se disponível)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("MEMÓRIA LIMPA")

    # Adição do ultimo peso do modelo que parou
    model = YOLO(model="Yolov11-WithRandomSamples/train7/weights/last.pt")  


    # Treinar
    results = model.train(
        data="cocoRandom.yaml",
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
        project = "Yolov11-WithRandomSamples",
        plots = True,
        resume=True,  # Continuar treinamento
        
        
    )