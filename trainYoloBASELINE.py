from ultralytics import YOLO
import os
import shutil
import torch
from pycocotools.coco import COCO
from tqdm import tqdm  # Para barra de progresso (opcional)

def preProcessing():

    # Diretórios
    DATA_DIR = "F:/COCO-Dataset/"
    os.makedirs(DATA_DIR, exist_ok=True)

    img_dir = os.path.join("F:/COCO-Dataset/", "CocoTrainFull/train2017")
    ann_file = os.path.join("F:/COCO-Dataset/","annotations2017/annotations2017/instances_train2017.json")

    # Carregar anotações
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()

    categories = coco.loadCats(coco.getCatIds())
    categories.sort(key=lambda x: x['id'])  # Ordenar por ID original

    # Mapear ID original → ID sequencial (0-79)
    coco_id_to_yolo_id = {cat['id']: idx for idx, cat in enumerate(categories)}
    print("Mapeamento de IDs:", coco_id_to_yolo_id)

    # Listar todas as categorias com IDs originais e YOLO
    for cat in categories:
        print(f"COCO ID: {cat['id']} → YOLO ID: {coco_id_to_yolo_id[cat['id']]} | Nome: {cat['name']}")

    train_label_dir = "F:/COCO-Dataset/CocoTrainFull/labels"

    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_file = img_info["file_name"]
        img_path = os.path.join(img_dir, img_file)
        
        # Gerar rótulo YOLO
        label_path = os.path.join(train_label_dir, img_file.replace(".jpg", ".txt"))
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)
        
        with open(label_path, "w") as f:
            for ann in annotations:
                if "bbox" not in ann or ann["area"] <= 0:
                    continue
                
                # Converter ID COCO para YOLO
                yolo_class_id = coco_id_to_yolo_id.get(ann['category_id'], -1)
                if yolo_class_id == -1:  # Pular classes não mapeadas
                    continue
                
                # Converter bbox
                x, y, w, h = ann["bbox"]
                img_width = img_info["width"]
                img_height = img_info["height"]
                
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                w_norm = w / img_width
                h_norm = h / img_height
                
                f.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

if __name__ == '__main__':

    # Limpeza do cache da GPU (se disponível)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("MEMÓRIA LIMPA")

    # Configuração do modelo (supondo que 'yolov11x' é a versão completa)
    model = YOLO(model="yolo11x.yaml")  #Sem pre-treinamento

    # Treinar
    results = model.train(
        data="cocoBaselinePath.yaml",
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
        project = "Yolov11-BASELINE",
        plots = True,
        
        
    )