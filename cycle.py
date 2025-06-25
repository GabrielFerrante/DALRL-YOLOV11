import os
import subprocess
import time
import json
import threading
import queue
import torch
import psutil
import gc
from datetime import datetime

# Configurações
CYCLES = 10
GPU_DEVICE = "0"  # ID da GPU
LOG_DIR = "active_learning_logs"
CHECKPOINT_FILE = os.path.join(LOG_DIR, "active_learning_checkpoint.json")
os.makedirs(LOG_DIR, exist_ok=True)

# Fila para impressão ordenada
print_queue = queue.Queue()

def save_checkpoint(cycle, script_idx):
    """Salva o ponto de retomada (checkpoint)"""
    checkpoint = {
        "cycle": cycle,
        "script_idx": script_idx,
        "timestamp": datetime.now().isoformat()
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)

def load_checkpoint():
    """Carrega o checkpoint se existir"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return None

def remove_checkpoint():
    """Remove o arquivo de checkpoint"""
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

def run_script(script_name, cycle, gpu_id=None):
    """Executa um script Python com log em tempo real"""
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    log_file = os.path.join(LOG_DIR, f"cycle_{cycle}_{os.path.splitext(script_name)[0]}.log")
    start_time = time.time()
    
    # Mensagem inicial
    print_queue.put(f"\n🚀 [Ciclo {cycle}] Iniciando: {script_name}")
    print_queue.put(f"⏰ Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        time.sleep(5)  # Pausa generosa
    
    try:
        # Iniciar processo
        process = subprocess.Popen(
            ["python", script_name, str(cycle)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Capturar saída em tempo real
        with open(log_file, 'w') as log_f:
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print_queue.put(f"[{script_name}] {output.strip()}")
                    log_f.write(output)
                    log_f.flush()
        
        return_code = process.poll()
        status = "success" if return_code == 0 else "failed"
        
    except Exception as e:
        return_code = -1
        status = "crashed"
        print_queue.put(f"❌ ERRO GRAVE: {str(e)}")
    
    duration = time.time() - start_time
    print_queue.put(f"🏁 [Ciclo {cycle}] {script_name} concluído")
    print_queue.put(f"⏱️ Duração: {duration/60:.2f} minutos")
    print_queue.put(f"📋 Log salvo em: {log_file}")
    print_queue.put(f"🔚 Status: {'✅ Sucesso' if status == 'success' else '❌ Falha'}")
    
    # Liberar recursos
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "cycle": cycle,
        "script": script_name,
        "status": status,
        "return_code": return_code,
        "duration": duration,
        "log_file": log_file,
        "timestamp": datetime.now().isoformat()
    }

def print_worker():
    """Trabalhador para impressão ordenada de logs"""
    while True:
        message = print_queue.get()
        if message is None:
            break
        print(message)
        print_queue.task_done()

def check_system_resources(min_ram=15, min_gpu=1.0):
    """Verifica se há recursos suficientes antes de executar um script"""
    ram_available = 100 - psutil.virtual_memory().percent
    gpu_available = 0
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        used_mem = torch.cuda.memory_allocated() / (1024 ** 3)
        gpu_available = total_mem - used_mem
    
    if ram_available < min_ram:
        print_queue.put(f"⚠️ RAM insuficiente: {ram_available:.1f}% livre < {min_ram}% requerido")
        return False
    
    if torch.cuda.is_available() and gpu_available < min_gpu:
        print_queue.put(f"⚠️ GPU insuficiente: {gpu_available:.1f}GB livre < {min_gpu}GB requerido")
        return False
    
    return True

def wait_for_resources(min_ram=15, min_gpu=1.0, max_wait=300, check_interval=30):
    """Aguarda até que os recursos estejam disponíveis"""
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if check_system_resources(min_ram, min_gpu):
            return True
        
        print_queue.put(f"⏳ Aguardando recursos... (RAM: >{min_ram}%, GPU: >{min_gpu}GB)")
        time.sleep(check_interval)
    
    print_queue.put(f"⌛ Tempo de espera excedido para recursos")
    return False

def active_learning_cycle(cycle, start_script_idx=0):
    """Executa um ciclo completo de Active Learning com gerenciamento de recursos"""
    print_queue.put(f"\n{'='*50}")
    print_queue.put(f"🔁 INICIANDO CICLO DE ACTIVE LEARNING #{cycle}")
    if start_script_idx > 0:
        print_queue.put(f"↩️ RETOMANDO EXECUÇÃO A PARTIR DO SCRIPT #{start_script_idx}")
    print_queue.put(f"{'='*50}")
    
    report = []
    scripts = [
        "agentTraining-WithRandomPool.py",
        "agentTraining-WithClusteringPool.py",
        "agentSelection-WithRandomPool.py",
        "agentSelection-WithClusteringPool.py",
        "trainYoloForRandom-Cycle.py",
        "trainYoloForClustering-Cycle.py"
    ]
    
    # Executar scripts a partir do ponto de retomada
    for idx in range(start_script_idx, len(scripts)):
        script = scripts[idx]
        
        # Verificar e aguardar recursos
        if not wait_for_resources(min_ram=15, min_gpu=1.0):
            print_queue.put(f"⏭️ Pulando {script} devido a recursos insuficientes")
            continue
        
        # Executar script
        result = run_script(script, cycle, GPU_DEVICE)
        report.append(result)
        
        # Se falhar, salvar checkpoint e interromper ciclo
        if result["status"] != "success":
            print_queue.put(f"⛔ Erro no script {script}! Salvando checkpoint...")
            save_checkpoint(cycle, idx)
            return report
        
        # Pausa entre scripts
        time.sleep(10)
    
    # Se completou o ciclo, remover checkpoint existente
    if start_script_idx > 0 or os.path.exists(CHECKPOINT_FILE):
        remove_checkpoint()
    
    print_queue.put(f"\n✅ CICLO {cycle} COMPLETO!")
    return report

def main():
    # Iniciar trabalhador de impressão
    printer_thread = threading.Thread(target=print_worker)
    printer_thread.start()
    
    execution_report = []
    checkpoint = load_checkpoint()
    start_cycle = 1
    start_script_idx = 0

    # Configurar ponto de partida com base no checkpoint
    if checkpoint:
        start_cycle = checkpoint["cycle"]
        start_script_idx = checkpoint["script_idx"]
        print_queue.put(f"🔍 Checkpoint encontrado: Ciclo {start_cycle}, Script {start_script_idx}")
        print_queue.put(f"🔄 Reiniciando execução a partir do ponto de falha")
    
    try:
        for cycle in range(start_cycle, CYCLES + 1):
            # Configurar índice inicial para o ciclo atual
            current_start_idx = start_script_idx if cycle == start_cycle else 0
            
            # Verificar temperatura da GPU
            if torch.cuda.is_available():
                if hasattr(torch.cuda, 'temperature'):
                    temp = torch.cuda.temperature()
                    if temp > 85:
                        print_queue.put(f"🔥 ALERTA: GPU quente ({temp}°C) - Aguardando resfriamento")
                        time.sleep(300)
            
            # Executar ciclo
            cycle_report = active_learning_cycle(cycle, current_start_idx)
            execution_report.extend(cycle_report)
            
            # Resetar índice de início após o primeiro ciclo retomado
            if cycle == start_cycle:
                start_script_idx = 0
            
            # Verificar se ciclo foi interrompido por falha
            if any(item["status"] != "success" for item in cycle_report):
                print_queue.put(f"⛔ Ciclo {cycle} interrompido devido a falha")
                break
            
            # Pausa entre ciclos
            time.sleep(30)
        
        # Salvar relatório final se completou todos os ciclos
        else:
            final_report = {
                "total_cycles": CYCLES,
                "execution_details": execution_report,
                "end_time": datetime.now().isoformat()
            }
            
            with open(os.path.join(LOG_DIR, "final_report.json"), "w") as f:
                json.dump(final_report, f, indent=2)
            
            print_queue.put("\n" + "="*50)
            print_queue.put(f"🏁 TODOS OS {CYCLES} CICLOS CONCLUÍDOS COM SUCESSO!")
            print_queue.put("="*50)
        
    except KeyboardInterrupt:
        print_queue.put("\n🛑 Execução interrompida pelo usuário!")
    
    except Exception as e:
        print_queue.put(f"\n❌ ERRO CRÍTICO: {str(e)}")
    
    finally:
        # Sinalizar para o trabalhador de impressão parar
        print_queue.put(None)
        printer_thread.join()
        
        # Limpeza final
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Verificar e instalar dependências necessárias
    dependencies = ["psutil", "torch"]
    for dep in dependencies:
        try:
            __import__(dep)
        except ImportError:
            print(f"Instalando {dep}...")
            subprocess.run(["pip", "install", dep])
    
    # Importar após possível instalação
    import torch
    import psutil
    
    main()