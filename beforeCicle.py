import os

scripts = ["trainYoloForRandom-Cicle.py",
           "trainYoloForClustering-Cicle.py"]
for script in scripts:
    os.system(f"python {script}") 