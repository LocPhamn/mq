import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, ToTensor
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from torch.nn.functional import interpolate
from pathlib import Path
import os
import PIL.Image as Image
import json
import random
import shutil
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
from skimage.exposure import match_histograms
from custom_module import  check_object_center_perpendicular, get_ground_bbox, get_ground_mask,check_ground_contact, match_color
import argparse


# Grounded SAM2 parameters
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swinb_cogcoor.pth"
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# model = torch.hub.load("isl-org/MiDaS", "DPT_Hybrid", pretrained=True)
# model.eval()
# transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

# # Build SAM2 predictor
# sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
# sam2_predictor = SAM2ImagePredictor(sam2_model)

# # Build Grounding DINO model
# grounding_model = load_model(
#     model_config_path=GROUNDING_DINO_CONFIG, 
#     model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
#     device=DEVICE
# )

"""
---- SINGLE LABEL --- 
`1.excavator. 2.bulldozer. 3.crane. 4.Mobile Crane. 5.dump truck. 6.construction truck. 7.forklift. 8.Scissors Lift. 9.Concrete mixer.

---- SUMMARY  LABEL ---
1.contruction truck 2. crane 3.earthmover 4.lifting vehicle
"""
classes = {
    0: ["concrete mixer", "construction truck","construction_truck", "dump truck", "truck"],
    1: ["crane", "mobile crane","scissors lift"],
    2: ["excavator", "bulldozer","earth_mover"],
    3: ["forklift","lifting_vehicle"],
}

idx_to_class = {v: k for k, values in classes.items() for v in values}

def labeling_grounding_dino(bg_path):
    text_prompt = "Grader. Road roller. Concrete mixer. excavator. bulldozer. crane. Mobile Crane. truck. top view dump truck. forklift. Scissors Lift."
    
    try:
        image_source, image = load_image(bg_path)
        # sam2_predictor.set_image(image_source)
        
        boxes, confidences, labels = predict(
            model=grounding_model,
            image=image,
            caption=text_prompt,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=DEVICE
        )
        
        if len(boxes) > 0:
            h, w, _ = image_source.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            
            # Make a writable copy of the image to avoid readonly array error
            image_draw = image_source.copy()
            
            for i, (box, label, confidence) in enumerate(zip(input_boxes, labels, confidences)):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_draw, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            labeled_image = Image.fromarray(cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB))
            labeled_image.save("labeled_image.png")
        else:
            print("No objects detected.")
    except Exception as e:
        print(f"Error during labeling: {e}")

def labeling_custom(bg_path,img_path, label, x1, y1, x2, y2):
    root = "generated_images/labels"
    bg_label = "labels/bg_images"           
    x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
    width, height = x2 - x1, y2 - y1
    os.makedirs(root, exist_ok=True)
    os.makedirs(bg_label, exist_ok=True)

    # base_name, idx = os.path.basename(bg_path).split('.png')[0].split(' ')
    base_name = os.path.basename(bg_path).split('.png')[0]
    img_name = os.path.basename(img_path).split('.')[0]
    label_name = os.path.basename(label).split('-')[0] 
    

    # bg_label_path = os.path.join(bg_label, f"{base_name} {idx}.txt")
    bg_label_path = os.path.join(bg_label, f"{base_name}.txt")
    copy_label_path = os.path.join(root, f"{img_name}.txt")

    print(f"Labeling {copy_label_path} with object from {label}")

    if not os.path.exists(bg_label_path):
        with open(bg_label_path, 'w') as f:
            f.write("")  # tạo file rỗng

    if not os.path.exists(copy_label_path):
        with open(bg_label_path, 'r') as f_src, open(copy_label_path, 'w') as f_dst:
            f_dst.write(f_src.read())

    with open(copy_label_path, 'a') as f:
        key = idx_to_class[label_name.lower()]

        file_size = f.tell()
        if key is not None:
            label = key
        else:
            print(f"Warning: Vehicle label '{label_name}' not found in classes dictionary.")
            return
        if file_size > 0:
            f.write(f"\n{label} {x_center} {y_center} {width} {height}")
        else:
            f.write(f"{label} {x_center} {y_center} {width} {height}")

def labeling_custom_stockbridge(bg_path,img_path, label, x1, y1, x2, y2):
    
    root = "generated_images/labels"        
    bg_label = "labels/bg_images/labels"           
    x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
    width, height = x2 - x1, y2 - y1
    os.makedirs(root, exist_ok=True)
    os.makedirs(bg_label, exist_ok=True)

    base_name = os.path.basename(bg_path).split('.jpg')[0]
    img_name = os.path.basename(img_path).split('.')[0]
    label_name = os.path.basename(label).split('-')[0] 
    

    copy_label_path = os.path.join(root, f"{img_name}.txt")

    print(f"Labeling {copy_label_path} ")

    # if not os.path.exists(bg_label_path):
    #     with open(bg_label_path, 'w') as f:
    #         f.write("")  # tạo file rỗng

    # if not os.path.exists(copy_label_path):
    #     with open(bg_label_path, 'r') as f_src, open(copy_label_path, 'w') as f_dst:
    #         f_dst.write(f_src.read())

    with open(copy_label_path, 'a') as f:
        file_size = f.tell()
        label = "3"
        if file_size > 0:
            f.write(f"\n{label} {x_center} {y_center} {width} {height}")
        else:
            f.write(f"{label} {x_center} {y_center} {width} {height}")

def test():
    img_path = "generated_images/images/out_7.png"
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Error: Could not load image from {img_path}")
        print("Please check if the file exists and the path is correct.")
        return

    label = "generated_images/labels/out_7.txt"

    if not os.path.exists(label):
        print(f"Error: Label file {label} does not exist.")
        return
    
    with open(label, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            parts = line.split()
            if len(parts) == 5:
                try:
                    class_id, x_center, y_center, w, h = parts
                    # Convert from YOLO format (center_x, center_y, width, height) to pixel coordinates
                    img_h, img_w = img.shape[:2]
                    x_center = float(x_center) * img_w
                    y_center = float(y_center) * img_h
                    width = float(w) * img_w
                    height = float(h) * img_h
                    
                    x1 = int(x_center - width/2)
                    y1 = int(y_center - height/2)
                    x2 = int(x_center + width/2)
                    y2 = int(y_center + height/2)
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"{class_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except ValueError as e:
                    print(f"Error parsing line: '{line}' - {e}")
                    continue
            else:
                print(f"Invalid line format: '{line}' (expected 5 parts, got {len(parts)})")
    
    cv2.imwrite("labeled_test.png", img)
    print("Labeled image saved as 'labeled_test.png'")
    
if __name__ == "__main__":
    # labeling_custom("images/bg_images/bg (19).png","generated_images/out1.png", "images/upview_vehicles/postprocess/crane_0.jpg", 0.1, 0.2, 0.3, 0.4)
    test()