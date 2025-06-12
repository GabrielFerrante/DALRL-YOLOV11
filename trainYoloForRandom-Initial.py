from ultralytics import YOLO



if __name__ == '__main__':
    # Configuração do modelo (supondo que 'yolov11x' é a versão completa)
    model = YOLO(model="yolo11x.yaml")  #Sem pre-treinamento

    # Hiperparâmetros customizados (opcional)
    model.args = {
        'epochs': 20,
        'batch': 8,
        'imgsz': 480,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'data': 'coco.yaml',
        'device' : 0,
    }

    # Treinar
    results = model.train(
        data="cocoRandom.yaml",
        epochs=25,
        imgsz=480,
        batch=8,
        cache=False,
        name="yolov11-initial-WithRandomSamples",
        pretrained=False,
        workers= 8,
        device = 0,
        dnn = True
        
    )