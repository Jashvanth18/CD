import cv2
import numpy as np
import sys
import os

# Add backend to path
sys.path.append('backend')

try:
    from inference import inspect_image
    print("Import successful")
    
    # Create dummy image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    print("Running inference...")
    vis, stage, label, dist = inspect_image(img)
    print("Inference successful")
    print(f"Stage: {stage}, Label: {label}, Dist: {dist}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
