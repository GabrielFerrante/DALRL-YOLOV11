import os

# ANTES, TREINE UMA VEZ OS MODELOS FORA DO CICLO.

scripts = ["agentTraining-WithRandomPool.py", 
           "agentTraining-WithClusteringPool.py", 
           "agentSelection-WithRandomPool.py", 
           "agentSelection-WithClusteringPool.py",
           "trainYoloForRandom-Cicle.py",
           "trainYoloForClustering-Cicle.py"]
for i in range(10):
    for script in scripts:
        os.system(f"python {script}") 