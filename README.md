# DALRL-YOLOV11
Faster Detection AL - YOLOV11 with based Reinforcement Deep Active Learning .

Workflow:
- Initialization:

    - Pre-trains YOLOv11 with a small set of labeled data.

    - Loads a large volume of unlabeled data (e.g. COCO dataset).

- Active Training Cycle:

    - Step 1 (Inference): YOLOv11 processes unlabeled images and calculates metrics (e.g. entropy, embeddings).

    - Step 2 (Selection via RL): The RL agent (e.g. PPO, DQN) analyzes the state (embeddings + uncertainty) and chooses the most useful images for labeling.

    - Step 3 (Labeling): The oracle labels the selected images.

    - Step 4 (Model Update): YOLOv11 is retrained with the new labeled data.

    - Step 5 (Reward): The RL agent receives a reward proportional to the improvement in YOLO (e.g. increase in mAP).

- Convergence:

    - The cycle is repeated until the model reaches the desired performance or the labeling resources are exhausted.
