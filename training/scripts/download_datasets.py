import os
from roboflow import Roboflow

ROBOFLOW_API_KEY = "TgOxTTC1807hdcjmegK0"  # fill this in

rf = Roboflow(api_key=ROBOFLOW_API_KEY)

print("📥 Downloading Worker-Safety...")
rf.workspace("computer-vision").project("worker-safety")\
  .version(1).download("yolov8", location=r"C:\industriguard_training\raw\worker_safety")

print("📥 Downloading bangga PPE-2...")
rf.workspace("bangga").project("ppe-2-ynh14")\
  .version(2).download("yolov8", location=r"C:\industriguard_training\raw\bangga")

print("✅ Roboflow datasets done")