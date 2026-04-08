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
from custom_module import  yolo_seg_to_mask, check_polygon_area, check_other_polygon_area
from labeling import labeling_custom, labeling_custom_stockbridge
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

def argument_parser():
    parser = argparse.ArgumentParser(description="Copy-Paste with Depth Estimation")
    parser.add_argument("--background_path",'-b', type=str, default="bg (14).png", help="Path to the background image")
    parser.add_argument("--output_dir","-o", type=str, default="generated_images/images", help="Directory to save generated images")
    parser.add_argument("--object_num", "-n", type=int, default=5, help="Number of objects to paste")
    parser.add_argument("--image_num", "-i", type=int, default=10, help="Number of images to generate")
    parser.add_argument("--match_color", "-m", type=str, default="n", help="match color or not (y/n)")

    args = parser.parse_args()
    return args

def segment_stockbridge_damper_with_grounded_sam2(image_source,mask):
    h, w = mask.shape
    if mask is not None: 
        rgba_img = np.zeros((h, w, 4), dtype=np.uint8)
        rgba_img[:, :, :3] = image_source  # RGB channels
        rgba_img[:, :, 3] = (mask * 255).astype(np.uint8)  # Alpha channel
        
        return rgba_img
    else:
        print("No stockbridge damper detected, returning full image")
        h, w, _ = image_source.shape
        rgba_img = np.zeros((h, w, 4), dtype=np.uint8)
        rgba_img[:, :, :3] = image_source  # RGB channels
        rgba_img[:, :, 3] = 255  # Full opacity
        
        return rgba_img
    
def compute_scale(depth_value, min_scale=1.0, max_scale=7.0):
    if isinstance(depth_value, tuple):
        depth_value = depth_value[0]
    scale = min_scale + (max_scale - min_scale) * float(depth_value)
    return scale

def minimum_object_size(obj_img, max_size=64):
    h, w = obj_img.shape[:2]
    scale = max_size / w
    new_w = max_size
    new_h = int(h * scale)

    resized_img = cv2.resize(obj_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return resized_img, resized_img.shape[:2]

def create_bbox(mask):
    # Chuyển về grayscale nếu là ảnh màu
    if len(mask.shape) == 3:
        if mask.shape[2] == 4:  # RGBA
            mask = mask[:, :, 3]
        else:  # RGB
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Tìm tọa độ các pixel khác 0
    if mask.dtype == np.float32 or mask.dtype == np.float64:
        rows, cols = np.where(mask > 0)
    else:
        rows, cols = np.where(mask > 0)  
    
    if len(rows) == 0 or len(cols) == 0:
        print("WARNING: No valid pixels found in mask!")
        return None
    
    x_min = int(cols.min())
    y_min = int(rows.min())
    x_max = int(cols.max())
    y_max = int(rows.max())
    
    return [x_min, y_min, x_max, y_max]

def mask_to_obb(mask):
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_uint8 = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = box.astype(int)
    x1,y1 = box[0]
    x2,y2 = box[1]
    x3,y3 = box[3]
    x4,y4 = box[2]

    top_coords = (x1, y1, x2, y2)
    bottom_coords = (x3, y3, x4, y4)
    
    return top_coords, bottom_coords

def interpolate_point(p1, p2, t):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    return p1 + t * (p2 - p1)

def visualize_transformed_label(oject_img, label_path, other_path):
    """Draw transformed polygon on final_img for verification"""
    img_vis = oject_img.copy()
    if len(img_vis.shape) == 2:
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)
    elif img_vis.shape[2] == 4:
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGBA2BGR)
        
    h, w = img_vis.shape[:2]
    
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            coords = list(map(float, parts[1:]))
            
            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * w)
                y = int(coords[i+1] * h)
                points.append([x, y])
            
            points = np.array(points, dtype=np.int32)
            cv2.polylines(img_vis, [points], True, (0, 255, 0), 2)
            
    if os.path.exists(other_path):
        with open(other_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                coords = list(map(float, parts[1:]))
                
                points = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * w)
                    y = int(coords[i+1] * h)
                    points.append([x, y])
                
                points = np.array(points, dtype=np.int32)
                cv2.polylines(img_vis, [points], True, (255, 0, 0), 2)
    return img_vis

def perpective_transformation(mask, image, bbox, rotation_type='random'):
    """
    Xoay ảnh theo 4 hướng cố định: left, right, up, down
    rotation_type: 'left', 'right', 'up', 'down', 'random'
    """
    xmin, ymin, xmax, ymax = bbox
    h, w = (ymax - ymin), (xmax - xmin)
    
    # Resize mask về kích thước nhỏ hơn
    scale = 1.0
    # inside_mask = cv2.resize(mask, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    inside_mask = mask
    inside_h, inside_w = inside_mask.shape[:2]
    scale_w = int(inside_w * scale)
    scale_h = int(inside_h * scale)
    
    cv2.imwrite("inside_mask.png", inside_mask)
    inside_h, inside_w = inside_mask.shape[:2]
    print(f"Image size: {w}x{h}")
    print(f"Inside mask size: {inside_w}x{inside_h}")
    
    # start_x, start_y = xmin + (w - inside_w) // 2, ymin + (h - inside_h) // 2
    # end_x, end_y = start_x + inside_w, start_y + inside_h
    
    start_x, start_y = xmin + (w - scale_w) // 2, ymin + (h - scale_h) // 2
    end_x, end_y = start_x + scale_w, start_y + scale_h
    
    rotate_scale = random.uniform(0.14, 0.2)

    # Define 4 rotation types với các điểm perspective trên inside_mask
    rotation_configs = {
        'left': {
            'p1': (start_x, start_y),
            'p2': (end_x, start_y + int(scale_h * rotate_scale)),
            'p3': (start_x, end_y- int(scale_h * rotate_scale)),
            'p4': (end_x, end_y )
        },
        'right': {
            'p1': (start_x, start_y + int(scale_h * rotate_scale)),
            'p2': (end_x, start_y),
            'p3': (start_x, end_y ),
            'p4': (end_x, end_y- int(scale_h * rotate_scale))
        },
        'up': {
            'p1': (start_x, start_y),
            'p2': (end_x, start_y),
            'p3': (start_x + int(scale_w * rotate_scale), end_y),
            'p4': (end_x - int(scale_w * rotate_scale), end_y)
        },
        'down': {
            'p1': (start_x + int(scale_w * rotate_scale), start_y),
            'p2': (end_x - int(scale_w * rotate_scale), start_y),
            'p3': (start_x, end_y),
            'p4': (end_x, end_y)
        },
    }
    
    # Random chọn 1 trong 4 hướng
    if rotation_type == 'random':
        rotation_type = random.choice(['left', 'right', 'up', 'down',])
    
    config = rotation_configs.get(rotation_type, rotation_configs['left'])
    
    p1 = config['p1']
    p2 = config['p2']
    p3 = config['p3']
    p4 = config['p4']
    
    # Visualize points for debugging (chuyển về tọa độ global để vẽ)
    image_debug = image.copy()
    
    p1_global = (p1[0], p1[1])
    p2_global = (p2[0], p2[1])
    p3_global = (p3[0], p3[1])
    p4_global = (p4[0], p4[1])
    
    cv2.circle(image_debug, p1_global, 15, (0, 255, 0), -1)
    cv2.circle(image_debug, p2_global, 15, (0, 0, 255), -1)
    cv2.circle(image_debug, p3_global, 15, (255, 0, 0), -1)
    cv2.circle(image_debug, p4_global, 15, (255, 255, 0), -1)
    
    cv2.putText(image_debug, 'p1', p1_global, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
    cv2.putText(image_debug, 'p2', p2_global, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
    cv2.putText(image_debug, 'p3', p3_global, cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)
    cv2.putText(image_debug, 'p4', p4_global, cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 2)
    
    cv2.rectangle(image_debug, (xmin, ymin), (xmax, ymax), (255, 0, 255), 2)
    cv2.rectangle(image_debug, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
    
    cv2.putText(image_debug, f'Type: {rotation_type}', (xmin, ymin - 10),cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 2)
    cv2.imwrite("points.png", image_debug)
    
    # Get mask dimensions
    mask_h, mask_w = mask.shape[:2]
    
    # Source points: 4 điểm perspective trên inside_mask (tọa độ local)
    src_pts = np.float32([p1, p2, p3, p4])
    
    # Destination points: 4 góc bbox gốc (trước khi scale) - tọa độ global
    dst_pts = np.float32([
        [xmin, ymin],     
        [xmax, ymin],        
        [xmin, ymax],        
        [xmax, ymax]           
    ])
    # Calculate perspective transform
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Apply transformation
    output = cv2.warpPerspective(inside_mask, H, (mask_w, mask_h))
    
    return output,H

def perpective_transformation_obb(mask, image, box, obb_coords, rotation_type='random'):
    """
    Xoay ảnh theo perspective transformation: chọn điểm trên OBB rồi kéo ra bounding box
    
    Args:
        mask: Alpha mask của object
        image: Background image
        obb_coords: Tuple of ((x1, y1, x2, y2), (x3, y3, x4, y4)) - top and bottom coordinates from mask_to_obb
        rotation_type: 'left', 'right', 'up', 'down', 'random'
    
    Returns:
        output: Transformed mask/image
    """
    
    xmin, ymin, xmax, ymax = box
    
    top_coords, bottom_coords = obb_coords
    x1, y1, x2, y2 = top_coords
    x3, y3, x4, y4 = bottom_coords
    
    # # Get mask dimensions
    mask_h, mask_w = mask.shape[:2]
    
    p1 = (x1, y1)
    p2 = (x2, y2)
    p3 = (x3, y3)
    p4 = (x4, y4)
    
    # set new point from p1 with vector (p1 pnew )//( p1 p2)
    p_new1 = interpolate_point(p1, p2, 0.3)
    p_new2 = interpolate_point(p2, p1, 0.3)
    p_new3 = interpolate_point(p3, p4, 0.3)
    p_new4 = interpolate_point(p4, p3, 0.3)
    

    # Source points: 4 điểm OBB
    src_pts = np.float32([p_new1, p_new2, p_new3, p_new4])
    
    dst_pts = np.float32([
        [xmin, ymin],     # top-left
        [xmax, ymin],     # top-right
        [xmin, ymax],     # bottom-left
        [xmax, ymax]      # bottom-right
    ])
    
    # Visualize points for debugging
    image_debug = image.copy()

    # Vẽ OBB gốc
    obb_points = np.array([(x1, y1), (x2, y2), (x4, y4), (x3, y3)])
    cv2.polylines(image_debug, [obb_points], isClosed=True, color=(255, 0, 255), thickness=2)
    
    # Vẽ bounding box đích
    cv2.rectangle(image_debug, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
    
    # Vẽ 4 điểm source
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]
    labels = ['p1', 'p2', 'p3', 'p4']
    for i, (pt, color, label) in enumerate(zip(src_pts, colors, labels)):
        cv2.circle(image_debug, (int(pt[0]), int(pt[1])), 5, color, -1)
        cv2.putText(image_debug, label, (int(pt[0]), int(pt[1])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 4, color, 2)
    
    # Vẽ 4 điểm destination
    for i, pt in enumerate(dst_pts):
        cv2.circle(image_debug, (int(pt[0]), int(pt[1])), 5, (255, 255, 255), 2)
    
    cv2.putText(image_debug, f'Type: {rotation_type}', (xmin, ymin - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite("points_obb.png", image_debug)
    
    # Calculate perspective transform
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    output = cv2.warpPerspective(mask, H, (mask_w, mask_h))
    
    return output

def transform_yolo_label_with_perspective(label_path, H, image_shape):
    """
    Transform YOLO segmentation labels using perspective matrix H
    
    Args:
        label_path: Path to original YOLO label file
        H: Perspective transformation matrix from cv2.getPerspectiveTransform
        image_shape: (height, width) of the transformed image
    
    Returns:
        transformed_label: String containing new YOLO format label
    """
    h, w = image_shape[:2]
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    transformed_labels = []
    
    for line in lines:
        parts = line.strip().split()
        class_id = parts[0]
        coords = list(map(float, parts[1:]))
        
        # Convert normalized coords to pixel coords
        points = []
        for i in range(0, len(coords), 2):
            x_norm, y_norm = coords[i], coords[i+1]
            x_pixel = x_norm * w
            y_pixel = y_norm * h
            points.append([x_pixel, y_pixel])
        
        # Apply perspective transformation
        points_array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed_points = cv2.perspectiveTransform(points_array, H)
        transformed_points = transformed_points.reshape(-1, 2)
        
        # Convert back to normalized coordinates
        normalized_coords = []
        for point in transformed_points:
            x_norm = point[0] / w
            y_norm = point[1] / h
            # Clamp to [0, 1]
            x_norm = max(0, min(1, x_norm))
            y_norm = max(0, min(1, y_norm))
            normalized_coords.extend([x_norm, y_norm])
        
        # Format as YOLO label
        label_str = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_coords])
        transformed_labels.append(label_str)
    
    return "\n".join(transformed_labels)

def save_transformed_label(original_label_path, H, image_shape, output_label_path):
    """
    Save transformed YOLO label to file
    
    Args:
        original_label_path: Path to original label file
        H: Perspective transformation matrix
        image_shape: Shape of transformed image
        output_label_path: Path to save new label file
    """
    transformed_label = transform_yolo_label_with_perspective(original_label_path, H, image_shape)
    
    with open(output_label_path, 'w') as f:
        f.write(transformed_label)
    
    print(f"Saved transformed label to: {output_label_path}")

def crop_after_perspective(rgba_img, padding=20):
    """
    Crop ảnh RGBA sau perspective transform để loại bỏ vùng đen thừa
    
    Args:
        rgba_img: RGBA image (numpy array) sau perspective transform
        padding: Padding around object (default=20)
        
    Returns:
        dict containing:
            - crop_rgba: Cropped RGBA image
            - crop_bbox: Bbox of cropped region [x1, y1, x2, y2]
            - crop_offset: Offset (x_offset, y_offset) để transform label
    """
    # Lấy alpha channel
    alpha = rgba_img[:, :, 3]
    
    # Tìm vùng có alpha > 0 (vùng có object)
    coords = np.column_stack(np.where(alpha > 0))
    
    if len(coords) == 0:
        # Không có object, return ảnh gốc
        return {
            'crop_rgba': rgba_img,
            'crop_bbox': [0, 0, rgba_img.shape[1], rgba_img.shape[0]],
            'crop_offset': (0, 0),
            'original_shape': rgba_img.shape
        }
    
    # Tìm bbox
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Add padding
    x1 = max(0, x_min - padding)
    y1 = max(0, y_min - padding)
    x2 = min(rgba_img.shape[1], x_max + padding)
    y2 = min(rgba_img.shape[0], y_max + padding)
    
    # Crop
    crop_rgba = rgba_img[y1:y2, x1:x2]
    
    return {
        'crop_rgba': crop_rgba,
        'crop_bbox': [x1, y1, x2, y2],
        'crop_offset': (x1, y1),
        'original_shape': rgba_img.shape
    }


def transform_label_after_crop(label_path, output_label_path, crop_offset, original_shape, crop_shape):
    """
    Transform label sau khi crop ảnh perspective
    
    Args:
        label_path: Path to label file (đã được transform bởi perspective)
        output_label_path: Path to save new label
        crop_offset: (x_offset, y_offset) - offset của vùng crop
        original_shape: Shape của ảnh trước khi crop
        crop_shape: Shape của ảnh sau khi crop
    """
    x_offset, y_offset = crop_offset
    orig_h, orig_w = original_shape[:2]
    crop_h, crop_w = crop_shape[:2]
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    transformed_labels = []
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
            
        class_id = parts[0]
        coords = list(map(float, parts[1:]))
        
        # Convert normalized coords to pixel coords (ảnh gốc)
        points = []
        for i in range(0, len(coords), 2):
            x_norm, y_norm = coords[i], coords[i+1]
            x_pixel = x_norm * orig_w
            y_pixel = y_norm * orig_h
            points.append([x_pixel, y_pixel])
        
        # Transform to crop coordinates
        crop_points = []
        for point in points:
            x_crop = point[0] - x_offset
            y_crop = point[1] - y_offset
            crop_points.append([x_crop, y_crop])
        
        # Convert back to normalized coordinates (ảnh crop)
        normalized_coords = []
        for point in crop_points:
            x_norm = point[0] / crop_w
            y_norm = point[1] / crop_h
            # Clamp to [0, 1]
            x_norm = max(0, min(1, x_norm))
            y_norm = max(0, min(1, y_norm))
            normalized_coords.extend([x_norm, y_norm])
        
        # Format as YOLO label
        label_str = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_coords])
        transformed_labels.append(label_str)
    
    with open(output_label_path, 'w') as f:
        f.write("\n".join(transformed_labels))

def simple_rotation_with_expansion(mask, image, bbox, rotation_angle='random', max_angle=15):
    """
    Xoay ảnh và tự động mở rộng canvas để không bị mất phần nào
    
    Args:
        mask: Alpha mask của object
        image: Background image (để visualize)
        bbox: [xmin, ymin, xmax, ymax]
        rotation_angle: 'random', 'left', 'right', 'up', 'down' hoặc số độ
        max_angle: Góc xoay tối đa khi random (default=15)
    
    Returns:
        rotated_mask: Ảnh sau khi xoay
        M: Affine transformation matrix (2x3)
    """
    xmin, ymin, xmax, ymax = bbox
    h, w = mask.shape[:2]
    
    # Map rotation types to angles
    rotation_map = {
        'left': max_angle,
        'right': -max_angle,
        'up': 0,
        'down': 180
    }
    
    # Random chọn góc nếu 'random'
    if rotation_angle == 'random':
        angle = random.uniform(-max_angle, max_angle)
    elif isinstance(rotation_angle, str):
        angle = rotation_map.get(rotation_angle, 0)
    else:
        angle = rotation_angle
    
    # Tính center của ảnh (không phải bbox)
    center = (w // 2, h // 2)
    
    # Tạo rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    
    # Tính kích thước canvas mới để chứa toàn bộ ảnh sau rotation
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Điều chỉnh translation để center ảnh
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    
    # Apply rotation
    rotated = cv2.warpAffine(mask, M, (new_w, new_h), 
                             flags=cv2.INTER_LINEAR, 
                             borderMode=cv2.BORDER_CONSTANT, 
                             borderValue=0)
    
    # Visualize
    image_debug = image.copy()
    cv2.rectangle(image_debug, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.circle(image_debug, center, 10, (0, 0, 255), -1)
    cv2.putText(image_debug, f'Angle: {angle:.1f}°', (xmin, ymin - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imwrite("rotation_debug.png", image_debug)
    
    return rotated, M

def transform_yolo_label_with_rotation(label_path, M, original_shape, rotated_shape):
    """
    Transform YOLO label với affine rotation matrix
    
    Args:
        label_path: Path to YOLO label file
        M: Affine transformation matrix (2x3) từ cv2.getRotationMatrix2D
        original_shape: (height, width) của ảnh gốc
        rotated_shape: (height, width) của ảnh sau rotation
    
    Returns:
        transformed_label: String chứa label mới
    """
    orig_h, orig_w = original_shape[:2]
    rot_h, rot_w = rotated_shape[:2]
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    transformed_labels = []
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
            
        class_id = parts[0]
        coords = list(map(float, parts[1:]))
        
        # Convert normalized coords to pixel coords (ảnh gốc)
        points = []
        for i in range(0, len(coords), 2):
            x_norm, y_norm = coords[i], coords[i+1]
            x_pixel = x_norm * orig_w
            y_pixel = y_norm * orig_h
            points.append([x_pixel, y_pixel])
        
        # Apply affine transformation
        points_array = np.array(points, dtype=np.float32)
        ones = np.ones((points_array.shape[0], 1), dtype=np.float32)
        points_homogeneous = np.hstack([points_array, ones])
        
        # Transform: M @ [x, y, 1]^T
        transformed_points = (M @ points_homogeneous.T).T
        
        # Convert back to normalized coordinates (ảnh sau rotation)
        normalized_coords = []
        for point in transformed_points:
            x_norm = point[0] / rot_w
            y_norm = point[1] / rot_h
            # Clamp to [0, 1]
            x_norm = max(0, min(1, x_norm))
            y_norm = max(0, min(1, y_norm))
            normalized_coords.extend([x_norm, y_norm])
        
        # Check if polygon is still valid (all points inside image)
        valid = all(0 <= normalized_coords[i] <= 1 for i in range(len(normalized_coords)))
        
        if not valid:
            print(f"Warning: Polygon class {class_id} out of bounds after rotation, skipping...")
            continue
        
        # Format as YOLO label
        label_str = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_coords])
        transformed_labels.append(label_str)
    
    return "\n".join(transformed_labels)

if __name__ == '__main__':
    args = argument_parser()
    alpha_folder = "alpha_images"
    countryside_folder = "images/countryside&city"
    out_dir = "new_images"
    
    image_out_dir = os.path.join(out_dir, "images")
    original_out_dir = os.path.join(out_dir, "original_images")     
    label_out_dir = os.path.join(out_dir, "labels_target")
    other_out_dir = os.path.join(out_dir, "labels_other")
    labeled_out_dir = os.path.join(out_dir, "labeled_images")
    log_dir = os.path.join(out_dir, "log")
    
    countryside_paths = [os.path.join(countryside_folder, f) for f in os.listdir(countryside_folder)]
    alpha_paths = []
    
    # shutil.rmtree(img_folder, ignore_errors=True)
    # shutil.rmtree(label_folder, ignore_errors=True)
    shutil.rmtree(alpha_folder, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    
    # os.makedirs(img_folder, exist_ok=True)
    # os.makedirs(label_folder, exist_ok=True)
    os.makedirs(alpha_folder, exist_ok=True)
    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)
    os.makedirs(other_out_dir, exist_ok=True)
    os.makedirs(original_out_dir, exist_ok=True)
    os.makedirs(labeled_out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    root = "dataset_28_11"
    normal_device_folder = os.path.join(root, "binh_thuong")
    damaged_device_folder = os.path.join(root, "loi")
    devices = ["cach_dien_polyme","chong_set_van","su_thuy_tinh","ta_chong_rung"]
    simple_rotation_devices = ["cach_dien_polyme", "chong_set_van"]

    random_rotate_choice = ['left', 'down']
    count = 0
    
    damaged_log_path = "damaged_device_log.txt"
    normal_log_path = "normal_device_log.txt"
    
    with open(os.path.join(log_dir,damaged_log_path), 'w') as f:
        f.write("Damaged device generation log\n")
    with open(os.path.join(log_dir,normal_log_path), 'w') as f:
        f.write("Normal device generation log\n")
        
    # Damaged device generation
    device_root = "train_gen_data_v3"
    device_folder = os.path.join(device_root, "loi")
    # shutil.rmtree(device_root, ignore_errors=True)
    os.makedirs(device_root, exist_ok=True)
    # os.makedirs(device_folder, exist_ok=True)

    for device in devices:
        image_folder = os.path.join(damaged_device_folder, f"{device}/images")
        label_folder = os.path.join(damaged_device_folder, f"{device}/labels_target")
        other_folder = os.path.join(damaged_device_folder, f"{device}/labels_other")
        
        device_image_folder = os.path.join(device_folder, f"{device}/images")
        device_label_folder = os.path.join(device_folder, f"{device}/labels_target")
        device_other_folder = os.path.join(device_folder, f"{device}/labels_other")
        os.makedirs(device_image_folder, exist_ok=True)
        os.makedirs(device_label_folder, exist_ok=True)
        os.makedirs(device_other_folder, exist_ok=True)
        
        for idx, path in enumerate(os.listdir(image_folder)):
            image_path = os.path.join(image_folder, path)
            base_image = path.replace(".jpg", "")
            base_label = path.replace(".jpg", ".txt")
            label_path = os.path.join(label_folder, base_label)
            other_path = os.path.join(other_folder, base_label)
            
            if not os.path.exists(label_path) or not os.path.exists(other_path):
                print(f"Label file or other label file not found, skipping...")
                continue
            
            image = cv2.imread(image_path)
            
            # Tạo mask và bbox từ ảnh gốc
            mask, pts = yolo_seg_to_mask(image_path, label_path)
            mask_other, pts_other = yolo_seg_to_mask(image_path, other_path)
            bbox = create_bbox(mask * 255)
            
            # Segment với ảnh gốc
            alpha_mask = segment_stockbridge_damper_with_grounded_sam2(image, mask)
            
            use_simple_rotation = device in simple_rotation_devices
            for choice in random_rotate_choice:
                # **BƯỚC 1: Áp dụng transformation**
                if use_simple_rotation:
                    rgba_img, M = simple_rotation_with_expansion(alpha_mask, image, bbox, choice, max_angle=15)
                else:
                    rgba_img, H = perpective_transformation(alpha_mask, image, bbox, choice)
                    M = H
                
                # **BƯỚC 2: Crop sau transform**
                crop_data = crop_after_perspective(rgba_img, padding=20)
                crop_rgba = crop_data['crop_rgba']
                crop_offset = crop_data['crop_offset']
                
                cv2.imwrite("bg_rgba.png", crop_rgba) 
                
                # **BƯỚC 3.1: Transform label lần 1 (rotation/perspective)**
                temp_label_path = f'temp_persp_label_{count}.txt'
                temp_other_path = f'temp_persp_other_{count}.txt'
                
                if use_simple_rotation:
                    # ✅ Transform với affine matrix
                    transformed_label = transform_yolo_label_with_rotation(
                        label_path, M, 
                        original_shape=image.shape,  # Shape gốc
                        rotated_shape=rgba_img.shape  # Shape sau rotation (CHƯA crop)
                    )
                    with open(temp_label_path, 'w') as f:
                        f.write(transformed_label)
                    
                    transformed_other = transform_yolo_label_with_rotation(
                        other_path, M,
                        original_shape=image.shape,
                        rotated_shape=rgba_img.shape
                    )
                    with open(temp_other_path, 'w') as f:
                        f.write(transformed_other)
                else:
                    # Transform với perspective matrix như cũ
                    save_transformed_label(label_path, H, rgba_img.shape, temp_label_path)
                    save_transformed_label(other_path, H, rgba_img.shape, temp_other_path)
                
                # **Bước 3.2: Check polygon area (trên ảnh đã transform nhưng CHƯA crop)**
                check_polygon_area(temp_label_path, 'area_log.txt', 
                                crop_offset, rgba_img.shape, crop_rgba.shape, 0.05)
                
                # **BƯỚC 4: Transform label lần 2 (sau crop)**
                output_label_path = f'{label_out_dir}/{choice}_{base_label}'
                output_other_path = f'{other_out_dir}/{choice}_{base_label}'
                
                output_label_path_device = f'{device_label_folder}/{choice}_{base_label}'
                output_other_path_device = f'{device_other_folder}/{choice}_{base_label}'
                
                transform_label_after_crop(
                    temp_label_path, output_label_path,
                    crop_offset, rgba_img.shape, crop_rgba.shape
                )
                transform_label_after_crop(
                    temp_other_path, output_other_path,
                    crop_offset, rgba_img.shape, crop_rgba.shape
                )
                
                transform_label_after_crop(
                    temp_label_path, output_label_path_device,
                    crop_offset, rgba_img.shape, crop_rgba.shape
                )
                transform_label_after_crop(
                    temp_other_path, output_other_path_device,
                    crop_offset, rgba_img.shape, crop_rgba.shape
                )
                
                # **BƯỚC 5: Composite với background**
                ojbect_img = Image.open("bg_rgba.png").convert("RGBA")  
                obj_w, obj_h = ojbect_img.size

                under_bg = random.choice(countryside_paths)
                background = Image.open(under_bg).convert("RGBA")
                bg_w, bg_h = background.size

                center_x = bg_w // 2
                center_y = bg_h // 2

                crop_x1 = max(0, center_x - obj_w // 2)
                crop_y1 = max(0, center_y - obj_h // 2)
                crop_x2 = min(bg_w, crop_x1 + obj_w)
                crop_y2 = min(bg_h, crop_y1 + obj_h)

                background_crop = background.crop((crop_x1, crop_y1, crop_x2, crop_y2))

                if background_crop.size != ojbect_img.size:
                    background_crop = background_crop.resize(ojbect_img.size, Image.Resampling.LANCZOS)

                final_img = Image.alpha_composite(background_crop, ojbect_img)
                
                # **BƯỚC 6: Visualize và lưu**
                cv_img = np.array(final_img)
                vis_img = visualize_transformed_label(cv_img, output_label_path, output_other_path)
                ori_img_labeled = visualize_transformed_label(image, label_path, other_path)
                
                cv2.imwrite(f'{labeled_out_dir}/{choice}_{base_image}.png', vis_img)
                
                pil_image = Image.fromarray(cv_img)
                pil_image.save(f'{image_out_dir}/{choice}_{base_image}.png')
                pil_image.save(f'{device_image_folder}/{choice}_{base_image}.png')
                
                cv2.imwrite(f'{original_out_dir}/{choice}_{base_image}.png', ori_img_labeled)
                cv2.imwrite(f"{original_out_dir}/ori_{count}.png", image)

                print("Đã tạo thư mục và lưu ảnh vào:", out_dir)
                
                with open(os.path.join(log_dir, damaged_log_path), 'a') as f:
                    f.write(f"ori_{count}_labeled <=> {image_path}\n")
                
                os.remove(temp_label_path)
                os.remove(temp_other_path)
                
                count += 1

    # Normal device generation
    device_folder = os.path.join(device_root, "binh_thuong")
    shutil.rmtree(device_folder, ignore_errors=True)
    os.makedirs(device_folder, exist_ok=True)

    for device in devices:
        image_folder = os.path.join(normal_device_folder, f"{device}/images")
        label_folder = os.path.join(normal_device_folder, f"{device}/labels_target")
        other_folder = os.path.join(normal_device_folder, f"{device}/labels_other")
        
        device_image_folder = os.path.join(device_folder, f"{device}/images")
        device_label_folder = os.path.join(device_folder, f"{device}/labels_target")
        device_other_folder = os.path.join(device_folder, f"{device}/labels_other")
        os.makedirs(device_image_folder, exist_ok=True)
        os.makedirs(device_label_folder, exist_ok=True)
        os.makedirs(device_other_folder, exist_ok=True)
        
        for idx, path in enumerate(os.listdir(image_folder)):
            image_path = os.path.join(image_folder, path)
            base_image = path.replace(".jpg", "")
            base_label = path.replace(".jpg", ".txt")
            label_path = os.path.join(label_folder, base_label)
            other_path = os.path.join(other_folder, base_label)
            
            if os.path.exists(label_path) == False:
                print(f"Label file not found or empty, skipping...")
                continue
            
            image = cv2.imread(image_path)
            image_copy = image.copy()

            mask, pts = yolo_seg_to_mask(image_path, label_path)
            bbox = create_bbox(mask * 255)
            
            alpha_mask = segment_stockbridge_damper_with_grounded_sam2(image_copy, mask)
            
            use_simple_rotation = device in simple_rotation_devices

            for choice in random_rotate_choice:
                # **BƯỚC 1: Áp dụng transformation**
                if use_simple_rotation:
                    rgba_img, M = simple_rotation_with_expansion(alpha_mask, image_copy, bbox, choice, max_angle=15)
                else:
                    rgba_img, H = perpective_transformation(alpha_mask, image_copy, bbox, choice)
                    M = H
                
                # ✅ **BƯỚC 2: CROP sau transform (QUAN TRỌNG!)**
                crop_data = crop_after_perspective(rgba_img, padding=20)
                crop_rgba = crop_data['crop_rgba']
                crop_offset = crop_data['crop_offset']
                
                cv2.imwrite("bg_rgba_crop.png", crop_rgba)  # ✅ Lưu ảnh crop
                
                # **BƯỚC 3.1: Transform label lần 1 (rotation/perspective)**
                temp_label_path = f'temp_normal_label_{count}.txt'
                temp_other_path = f'temp_normal_other_{count}.txt'
                
                if use_simple_rotation:
                    transformed_label = transform_yolo_label_with_rotation(
                        label_path, M,
                        original_shape=image_copy.shape,
                        rotated_shape=rgba_img.shape  # Shape TRƯỚC crop
                    )
                    with open(temp_label_path, 'w') as f:
                        f.write(transformed_label)
                else:
                    save_transformed_label(label_path, H, rgba_img.shape, temp_label_path)
                
                # Tạo file other rỗng
                with open(temp_other_path, 'w') as f:
                    f.write("")
                
                # ✅ **BƯỚC 3.2: Transform label lần 2 (sau crop)**
                device_output_label_path = f'{device_label_folder}/{choice}_{base_label}'
                device_output_other_path = f'{device_other_folder}/{choice}_{base_label}'
                
                output_label_path = f'{label_out_dir}/out_{count}.txt'
                output_other_path = f'{other_out_dir}/other_{count}.txt'
                
                transform_label_after_crop(
                    temp_label_path, device_output_label_path,
                    crop_offset, rgba_img.shape, crop_rgba.shape
                )
                transform_label_after_crop(
                    temp_label_path, output_label_path,
                    crop_offset, rgba_img.shape, crop_rgba.shape
                )
                
                transform_label_after_crop(
                    temp_other_path, device_output_other_path,
                    crop_offset, rgba_img.shape, crop_rgba.shape
                )
                transform_label_after_crop(
                    temp_other_path, output_other_path,
                    crop_offset, rgba_img.shape, crop_rgba.shape
                )
                
                # **BƯỚC 4: Composite với background**
                ojbect_img = Image.open("bg_rgba_crop.png").convert("RGBA")  # ✅ Load ảnh CROP
                obj_w, obj_h = ojbect_img.size

                under_bg = random.choice(countryside_paths)
                background = Image.open(under_bg).convert("RGBA")
                bg_w, bg_h = background.size

                center_x = bg_w // 2
                center_y = bg_h // 2

                crop_x1 = max(0, center_x - obj_w // 2)
                crop_y1 = max(0, center_y - obj_h // 2)
                crop_x2 = min(bg_w, crop_x1 + obj_w)
                crop_y2 = min(bg_h, crop_y1 + obj_h)

                background_crop = background.crop((crop_x1, crop_y1, crop_x2, crop_y2))

                if background_crop.size != ojbect_img.size:
                    background_crop = background_crop.resize(ojbect_img.size, Image.Resampling.LANCZOS)

                final_img = Image.alpha_composite(background_crop, ojbect_img)
                
                cv_img = np.array(final_img)
                vis_img = visualize_transformed_label(cv_img, output_label_path, output_other_path)
                ori_img_labeled = visualize_transformed_label(image_copy, label_path, other_path)
                
                cv2.imwrite(f'{labeled_out_dir}/out_{count}_labeled.png', vis_img)
                
                pil_image = Image.fromarray(cv_img)
                pil_image.save(f'{image_out_dir}/out_{count}.png')
                pil_image.save(f'{device_image_folder}/{choice}_{base_image}.png')
                
                cv2.imwrite(f'{original_out_dir}/ori_{count}_labeled.png', ori_img_labeled)
                cv2.imwrite(f"{original_out_dir}/ori_{count}.png", image_copy)

                print("Đã tạo thư mục và lưu ảnh vào:", out_dir)
                
                with open(os.path.join(log_dir, normal_log_path), 'a') as f:
                    f.write(f"ori_{count}_labeled <=> {image_path}\n")
                
                # ✅ Cleanup temp files
                os.remove(temp_label_path)
                os.remove(temp_other_path)
                
                count += 1

