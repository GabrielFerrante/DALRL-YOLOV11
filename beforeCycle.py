import os
import subprocess
import time
import threading
import json
from datetime import datetime

# Configurações
GPU_DEVICE = "0"  # ID da GPU
LOG_DIR = "yolo_initial_training_logs"
os.makedirs(LOG_DIR, exist_ok=True)

def run_script(script_name, gpu_id=None):
    """Executa um script Python com controle de GPU e logging"""
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"{timestamp}_{os.path.splitext(script_name)[0]}.log")
    
    start_time = time.time()
    print(f"\n🚀 Iniciando: {script_name}")
    print(f"⏰ Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        result = subprocess.run(
            ["python", script_name],
            env=env,
            check=True,
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT
        )
        status = "success"
    except subprocess.CalledProcessError as e:
        status = "failed"
        print(f"❌ Erro em {script_name} - Código {e.returncode}")
    
    duration = time.time() - start_time
    print(f"🏁 Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️ Duração: {duration/60:.2f} minutos")
    print(f"📋 Log salvo em: {log_file}")
    
    return {
        "script": script_name,
        "status": status,
        "duration": duration,
        "log_file": log_file,
        "timestamp": datetime.now().isoformat()
    }

def run_parallel(scripts, gpu_id=None):
    """Executa scripts em paralelo usando threads"""
    threads = []
    results = []
    
    # Função wrapper para armazenar resultados
    def script_runner(script):
        result = run_script(script, gpu_id)
        results.append(result)
    
    # Iniciar threads
    for script in scripts:
        t = threading.Thread(target=script_runner, args=(script,))
        t.start()
        threads.append(t)
        time.sleep(10)  # Espaço entre inícios para evitar conflitos
    
    # Aguardar término
    for t in threads:
        t.join()
    
    return results

def main():
    scripts = [
        "trainYoloForRandom-Cicle.py",
        "trainYoloForClustering-Cicle.py"
    ]
    
    print(f"\n{'='*50}")
    print(f"🚀 INICIANDO TREINOS YOLO PARALELOS")
    print(f"{'='*50}")
    
    # Opção 1: Execução sequencial (mais seguro para GPU com pouca memória)
    results = [run_script(script, GPU_DEVICE) for script in scripts]
    
    # Opção 2: Execução paralela (mais rápido se a GPU tiver recursos)
    #results = run_parallel(scripts, GPU_DEVICE)
    
    # Salvar relatório
    report_file = os.path.join(LOG_DIR, "training_report.json")
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Verificar resultados
    success = all(r['status'] == 'success' for r in results)
    
    print("\n" + "="*50)
    print(f"📊 RELATÓRIO FINAL:")
    for r in results:
        print(f"- {r['script']}: {'✅' if r['status'] == 'success' else '❌'} ({r['duration']/60:.1f} min)")
    
    if success:
        print(f"\n🏁 TODOS OS TREINOS CONCLUÍDOS COM SUCESSO!")
    else:
        print(f"\n⚠️ ALGUNS TREINOS FALHARAM! Verifique os logs.")
    print("="*50)

if __name__ == "__main__":
    main()