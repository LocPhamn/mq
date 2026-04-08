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
from custom_module import  *
from labeling import labeling_custom
import argparse
from glob import glob
from copy_paster_config import *
from image_process import enhance_edges
import requests

# Grounded SAM2 parameters
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swinb_cogcoor.pth"
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load("isl-org/MiDaS", "DPT_Hybrid", pretrained=True)
model.eval()
transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

# Build SAM2 predictor
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# Build Grounding DINO model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

def argument_parser():
    parser = argparse.ArgumentParser(description="Copy-Paste with Depth Estimation")
    parser.add_argument("--background_path",'-b', type=str, default="bg (14).png", help="Path to the background image")
    parser.add_argument("--output_dir","-o", type=str, default="generated_images/images", help="Directory to save generated images")
    parser.add_argument("--object_num", "-n", type=int, default=5, help="Number of objects to paste")
    parser.add_argument("--image_num", "-i", type=int, default=10, help="Number of images to generate")
    parser.add_argument("--match_color", "-m", type=str, default="n", help="match color or not (y/n)")
    parser.add_argument("--send_hunyuan", "-s", type=str, default="n", help="send to hunyuan3d or not (y/n)")
    parser.add_argument("--hunyuan_url", type=str, default=os.environ.get("HUNYUAN3D_URL", "http://hunyuan3d:8000"),
                        help="Hunyuan3D API base URL")
    parser.add_argument("--hunyuan_output_dir", type=str, default="generated_images/3d", help="Output dir for 3D outputs")
    parser.add_argument("--visualize", "-v", type=str, default="n", help="Visualize the results or not (y/n)")

    args = parser.parse_args()
    return args

def crop_vehicle(image,idx):
    text_prompt = "excavator. bulldozer. crane. Mobile Crane. dump truck. construction truck. forklift. Scissors Lift. Concrete mixer."
    image_path = image
    array_image = cv2.imread(image)
    image_source, image = load_image(image)
    
    sam2_predictor.set_image(image_source)
    
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
        for i, (box, label) in enumerate(zip(input_boxes,labels)):
            x1, y1, x2, y2 = map(int, box)
            cropped_img = array_image[y1:y2, x1:x2]
            cropped_img,_ = minimum_object_size(cropped_img, max_size= 256)
            cv2.imwrite(f"images/upview_vehicles/postprocess/{label}{idx}-{i}.jpg", cropped_img)
            print(f"Saved cropped vehicle: images/upview_vehicles/postprocess/{label}{idx}-{i}.jpg")

        # os.remove(image_path)

    
def estimate_depth(img):
    input_batch = transform(img)
    with torch.no_grad():
        prediction = model(input_batch)
        depth = prediction.squeeze().cpu().numpy()
    return depth / depth.max() 

def segment_vehicle_with_grounded_sam2(cropped_img):
    text_prompt = "construction vehicle. excavator. bulldozer. crane. Mobile Crane. dump truck. construction truck. forklift. Scissors Lift. top view dump truck. Concrete mixer."
    
    if len(cropped_img.shape) == 3:
        if cropped_img.shape[2] == 3:  # BGR
            image_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        else:  
            image_rgb = cropped_img
    else:
        image_rgb = cropped_img
    
    pil_image = Image.fromarray(image_rgb)
    
    temp_path = "temp_vehicle_crop.jpg"
    pil_image.save(temp_path)
    
    try:
        image_source, image = load_image(temp_path)
        sam2_predictor.set_image(image_source)
        
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
            
            masks, scores, logits = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            
            # Tạo RGBA image với RGB channels đúng
            rgba_img = np.zeros((h, w, 4), dtype=np.uint8)
            rgba_img[:, :, :3] = image_source  # RGB channels
            rgba_img[:, :, 3] = (best_mask * 255).astype(np.uint8)  # Alpha channel
            
            return rgba_img, best_mask
        else:
            print("No vehicle detected, returning None")
            return None, None
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def remove_vehicle_background(cropped_img):
    rgba_img, mask = segment_vehicle_with_grounded_sam2(cropped_img)
    if rgba_img is None:
        return None
    return rgba_img

def get_region_depth(depth_map, box, ground_mask=None):
    """
    Tính depth value trung bình của vùng box, chỉ tính trên các pixel là ground
    
    Args:
        depth_map: depth map của ảnh background
        box: [x1, y1, x2, y2] bounding box
        ground_mask: binary mask (0 hoặc 1) cho ground, nếu None thì tính toàn bộ box
    
    Returns:
        mean_depth: giá trị depth trung bình
        ground_pixel_count: số lượng pixel là ground trong box
    """
    x1, y1, x2, y2 = [int(x) for x in box]
    
    depth_region = depth_map[y1:y2, x1:x2]
    
    if ground_mask is not None:
        mask_region = ground_mask[y1:y2, x1:x2]
        
        ground_pixel_count = np.sum(mask_region == 1)
        
        if ground_pixel_count == 0:
            print("WARNING: No ground pixels found in box region!")
            # Fallback: tính toàn bộ vùng
            mean_depth = np.mean(depth_region)
            return mean_depth
        
        ground_depth_values = depth_region[mask_region == 1]
        mean_depth = np.mean(ground_depth_values)
                
        print(f"Mean depth (ground only): {mean_depth:.3f}")
        
        return mean_depth
    else:
        mean_depth = np.mean(depth_region)
        total_pixels = depth_region.size
        print(f"Region mean (all pixels): {mean_depth:.3f}")
        print(f"Total pixels in region: {total_pixels}")
        return mean_depth
    
def compute_scale(depth_value, min_scale=2.0, max_scale=7.0):
    if isinstance(depth_value, tuple):
        depth_value = depth_value[0]
    scale = min_scale + (max_scale - min_scale) * float(depth_value)
    return scale

def minimum_object_size(obj_img, max_size):
    target_area = max_size * max_size
    h, w = obj_img.shape[:2]
    current_area = h * w
    scale = np.sqrt(target_area / current_area)
    
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized_img = cv2.resize(obj_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return resized_img, resized_img.shape[:2]

def paste_object_with_alpha(bg_path, bg_img, obj_rgba,alpha_path, ground_polygon, depth_map, mask, colored, boxes, triangle_pts):
    """ Params:
        bg_img: ảnh background (PIL Image)
        obj_rgba: ảnh object với kênh alpha (numpy array)
        ground_polygon: polygon ground để paste object vào [[x1,y1], [x2,y2], ...]
        depth_map: depth map của ảnh background (numpy array)
        mask: binary mask (0 hoặc 1) cho ground (numpy array)
        colored: "y" or "n", có match color hay không
    Returns:
        bg_img: ảnh background sau khi paste object (PIL Image)
        final_bbox: bounding box của object đã paste [x1, y1, x2, y2]
        
        """
    bg_img = np.array(bg_img)
    obj_rgba, (h, w) = minimum_object_size(obj_rgba, max_size=MINIMUM_IMAGE_SIZE)
    rows, cols = np.where(mask == 1)
    points = np.column_stack((cols, rows))
    alpha_name = os.path.basename(alpha_path)
    full_path = "alpha_full_images/"+alpha_name
    
    
    idx = None
    bl_point = None
    
    if len(points) > 0:
        idx = np.random.choice(len(points),1, replace=False)
        bl_point = points[idx][0]
        
    if bl_point is None:
        print("No ground points found in mask, cannot paste object.")
        return bg_img, (0,0,0,0)
    
    bg_height, bg_width = bg_img.shape[:2]
    
    # Vòng lặp tìm vị trí phù hợp
    while True:
        x1_offset, y2_offset = bl_point
        x2_offset = x1_offset + w
        y1_offset = y2_offset - h

        paste_bbox = [x1_offset, y1_offset, x2_offset, y2_offset]
        depth_val = get_region_depth(depth_map, paste_bbox, mask)
        scale = compute_scale(depth_value=depth_val)
               
        if np.isnan(scale) or np.isinf(scale) or scale <= 0:
            scale = 1.0
        
        new_size = (int(w * scale), int(h * scale))
        scaled_w, scaled_h = new_size
        
        final_y2 = y2_offset  
        final_y1 = final_y2 - scaled_h  
        final_x1 = x1_offset
        final_x2 = final_x1 + scaled_w
        
        final_x1 = max(0, final_x1)
        final_x2 = min(final_x2, bg_width)
        final_y1 = max(0, final_y1)
        final_y2 = min(final_y2, bg_height)
        
        paste_start_y = int(final_y1)
        paste_end_y = int(final_y2)
        paste_start_x = int(final_x1)
        paste_end_x = int(final_x2)
        
        final_bbox = [paste_start_x, paste_start_y, paste_end_x, paste_end_y]
        
        bottom_left_in = point_in_polygon(paste_start_x, paste_end_y, ground_polygon)
        bottom_right_in = point_in_polygon(paste_end_x, paste_end_y, ground_polygon)
        inside_polygon = bottom_left_in and bottom_right_in
        
        if (check_object_iou_with_each_other(final_bbox, boxes, BOUDING_BOX_THRESHOLD) and \
            check_bottom_line_on_ground(mask, paste_start_x, paste_end_y, paste_end_x) and \
            inside_polygon):
            break
            
        idx = np.random.choice(len(points), 1, replace=False)
        bl_point = points[idx][0]
        print("points is not suitable, reselecting...")
    
    print(f"Paste region: [{paste_start_x}, {paste_start_y}, {paste_end_x}, {paste_end_y}]")
    print(f"Depth value at paste region: {depth_val:.3f}")
    print(f"Computed scale: {scale:.3f}")
    print(f"Final object size: {scaled_w}x{scaled_h}")
    print("----------------------------------------------------------")

    obj_resized = cv2.resize(obj_rgba, new_size, interpolation=cv2.INTER_CUBIC)
    sharp_obj = enhance_edges(obj_resized)
    obj_resized = cv2.cvtColor(sharp_obj, cv2.COLOR_BGRA2RGBA)
    
    # Tìm polygon từ alpha mask của object
    obj_polygon, obj_contour = get_polygon_from_alpha_mask(obj_resized, num_points_range=(4, 6))
    
    alpha_layer = np.zeros((bg_height, bg_width, 4), dtype=np.uint8)
    
    actual_paste_h = paste_end_y - paste_start_y
    actual_paste_w = paste_end_x - paste_start_x
    obj_h, obj_w = obj_resized.shape[:2]
    
    if obj_h != actual_paste_h or obj_w != actual_paste_w:
        obj_resized = obj_resized[:actual_paste_h, :actual_paste_w]
        obj_h, obj_w = actual_paste_h, actual_paste_w
    
    depth_crop = depth_map[paste_start_y:paste_end_y, paste_start_x:paste_end_x]
    
    if depth_crop.shape[0] > 0 and depth_crop.shape[1] > 0:
        depth_resized = cv2.resize(depth_crop, (obj_w, obj_h), 
                                   interpolation=cv2.INTER_LINEAR)
        
        # Apply occlusion mask
        occlusion_mask = depth_resized > depth_val        
        obj_resized[:, :, 3][occlusion_mask] = 0
    
    # Paste object_resized lên alpha_layer tại tọa độ (paste_start_x, paste_start_y)
    alpha_layer[paste_start_y:paste_end_y, paste_start_x:paste_end_x] = obj_resized
    # Convert sang PIL để xử lý
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
    bg_img_pil = Image.fromarray(bg_img)
    alpha_layer_pil = Image.fromarray(alpha_layer)
    alpha_layer_pil.save(full_path)
    
    # Apply color matching nếu cần
    if colored == "y":
        mean_bg_color = get_mean_color(bg_img, final_bbox)
        obj_crop = alpha_layer_pil.crop((paste_start_x, paste_start_y, paste_end_x, paste_end_y))
        obj_crop_blended = color_transfer_blend(obj_crop, bg_img_pil, mean_bg_color, alpha=0.5)
        alpha_layer_array = np.array(alpha_layer_pil)
        alpha_layer_array[paste_start_y:paste_end_y, paste_start_x:paste_end_x] = np.array(obj_crop_blended)
        alpha_layer_pil = Image.fromarray(alpha_layer_array)
    
    # Paste toàn bộ alpha_layer vào background
    bg_img_pil.paste(alpha_layer_pil, (0, 0), alpha_layer_pil)
    
    # Vẽ polygon của object lên background (nếu tìm được)
    adjusted_polygon = None
    
    if obj_polygon is not None:
        # Adjust polygon coordinates theo vị trí paste
        adjusted_polygon = obj_polygon.copy()
        adjusted_polygon[:, 0] += paste_start_x
        adjusted_polygon[:, 1] += paste_start_y
        
        # Convert PIL sang numpy để vẽ polygon
        # bg_array = np.array(bg_img_pil)
        # cv2.polylines(bg_array, [adjusted_polygon.astype(np.int32)], 
        #               isClosed=True, color=(255, 0, 0), thickness=2)
        
        # # Vẽ các điểm đỉnh của polygon
        # for i, point in enumerate(adjusted_polygon):
        #     cv2.circle(bg_array, tuple(point.astype(int)), 4, (0, 255, 255), -1)
        #     cv2.putText(bg_array, f'A{i}', tuple(point.astype(int) + 5), 
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Convert lại sang PIL
        # bg_img_pil = Image.fromarray(bg_array)

    return bg_img_pil, (paste_start_x, paste_start_y, paste_end_x, paste_end_y), adjusted_polygon

def find_smallest_angle_point(polygon):
   
    n = len(polygon)
    min_angle = float('inf')
    min_angle_idx = 0
    
    for i in range(n):
        # 3 điểm liên tiếp
        p1 = polygon[(i - 1) % n]  # điểm trước
        p2 = polygon[i]             # điểm hiện tại
        p3 = polygon[(i + 1) % n]  # điểm sau
        
        # Vector từ p2 đến p1 và p2 đến p3
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        # Tính góc giữa 2 vector
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 > 0 and norm_v2 > 0:
            cos_angle = dot_product / (norm_v1 * norm_v2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Tránh lỗi numerical
            angle = np.arccos(cos_angle)
            angle_deg = np.degrees(angle)
            
            if angle_deg < min_angle:
                min_angle = angle_deg
                min_angle_idx = i
    
    # Lấy 2 cạnh kề của điểm có góc nhỏ nhất
    p_prev = polygon[(min_angle_idx - 1) % n]
    p_curr = polygon[min_angle_idx]
    p_next = polygon[(min_angle_idx + 1) % n]
    
    edge1 = [p_prev, p_curr]
    edge2 = [p_curr, p_next]
    
    return min_angle_idx, min_angle, edge1, edge2

def get_polygon_from_alpha_mask(alpha_img, num_points_range=(4, 6)):
    if alpha_img.shape[2] == 4:
        alpha_channel = alpha_img[:, :, 3]
    else:
        alpha_channel = cv2.cvtColor(alpha_img, cv2.COLOR_RGB2GRAY)
    
    _, binary_mask = cv2.threshold(alpha_channel, 10, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if len(contours) == 0:
        print("No contours found in alpha mask")
        return None, None
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    hull = cv2.convexHull(largest_contour)
    
    min_points, max_points = num_points_range
    epsilon_start = 0.01
    epsilon_max = 0.1
    epsilon_step = 0.005
    
    polygon = None
    epsilon = epsilon_start
    
    while epsilon <= epsilon_max:
        perimeter = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon * perimeter, True)
        num_points = len(approx)
        
        if min_points <= num_points <= max_points:
            polygon = approx.reshape(-1, 2)
            break
        
        if num_points < min_points:
            epsilon -= epsilon_step
            if epsilon < 0.001:
                epsilon = 0.001
            perimeter = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon * perimeter, True)
            polygon = approx.reshape(-1, 2)
            print(f"Using {len(polygon)} points (minimum epsilon reached)")
            break
        
        epsilon += epsilon_step
    
    if polygon is None:
        # Fallback: dùng approximation với epsilon cố định
        perimeter = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
        polygon = approx.reshape(-1, 2)
        print(f"Fallback: using polygon with {len(polygon)} points")
    
    return polygon, largest_contour

def get_rotate_degree(vanishing_point, base_midpoint, lines):
    vp = np.array(vanishing_point, dtype=np.float32)
    mid = np.array(base_midpoint, dtype=np.float32)
    x1, y1 = lines[0]
    x2, y2 = lines[1]
    
    direction_vector = mid - vp
    direction_vector /= np.linalg.norm(direction_vector)
    h,w = y2-y1, x2-x1
    center_pts = (int(lines[0][0]+w/2), int(lines[0][1]+h/2))
    bottome_pts = (int(lines[0][0]+w/2), lines[1][1])
    
    line_vec = np.array(center_pts, dtype=np.float32) - np.array(bottome_pts, dtype=np.float32)
    line_vec /= np.linalg.norm(line_vec)
    print("all points: vanishing_point {}, base_midpoint {}, lines {}".format(vanishing_point, base_midpoint, lines))
    
    print("Direction vector:", direction_vector)
    print("Line vector:", line_vec)
    
    dot = np.dot(direction_vector, line_vec)
    dot = np.clip(dot, -1.0, 1.0)
    angle = np.arccos(dot)

    angle_degrees = np.degrees(angle)
    return angle_degrees

def get_rotate_degree_cross_line(vanishing_point, base_midpoint, lines):
    vp = np.array(vanishing_point, dtype=np.float32)
    mid = np.array(base_midpoint, dtype=np.float32)
    x1, y1 = lines[0]
    x2, y2 = lines[1]
    
    direction_vector = mid - vp
    direction_vector /= np.linalg.norm(direction_vector)
    
    line_vec = np.array(lines[1], dtype=np.float32) - np.array(lines[0], dtype=np.float32)
    line_vec /= np.linalg.norm(line_vec)
    print("all points: vanishing_point {}, base_midpoint {}, lines {}".format(vanishing_point, base_midpoint, lines))
    
    print("Direction vector:", direction_vector)
    print("Line vector:", line_vec)
    
    dot = np.dot(direction_vector, line_vec)
    dot = np.clip(dot, -1.0, 1.0)
    angle = np.arccos(dot)

    angle_degrees = np.degrees(angle)
    return angle_degrees

def normalize_yaw(angle_deg):
    return angle_deg % 360.0


def wait_for_file(file_path, timeout=300, poll_interval=2.0):
    """
    Chờ cho đến khi file xuất hiện và có size > 0.
    
    Args:
        file_path: đường dẫn file cần chờ
        timeout: số giây tối đa chờ (mặc định 300s)
        poll_interval: khoảng thời gian giữa các lần kiểm tra (giây)
    
    Returns:
        True nếu file xuất hiện đúng hạn, False nếu timeout
    """
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            print(f"[WAIT] File ready: {file_path}")
            return True
        remaining = int(deadline - time.time())
        print(f"[WAIT] Waiting for {file_path} ... ({remaining}s remaining)")
        time.sleep(poll_interval)
    print(f"[WAIT] Timeout! File not found after {timeout}s: {file_path}")
    return False


def send_to_hunyuan3d(image_path, yaw_deg, hunyuan_url, output_dir, pitch_deg=0, turn=1):
    os.makedirs(output_dir, exist_ok=True)
    image_path_abs = os.path.abspath(image_path)
    image_name = os.path.basename(image_path)
    payload = {
        "image_path": image_path_abs,
        "image_name": image_name,
        "yaw_deg": yaw_deg,
        "pitch_deg": float(pitch_deg),
        "turn": turn,
        "img_size": 640,
    }
    try:
        response = requests.post(
            f"{hunyuan_url.rstrip('/')}/generate_view",
            json=payload,
            timeout=600,
        )
        if not response.ok:
            print(f"Hunyuan3D error response: {response.status_code} {response.text}")
            response.raise_for_status()
        data = response.json()
        print(f"Hunyuan3D response: {data}")
        return data
    except Exception as exc:
        print(f"Hunyuan3D request failed: {exc}")
        return None
    

if __name__ == '__main__':
    args = argument_parser()
    img_folder = args.output_dir
    label_folder = "generated_images/labels"
    img_bg_path = f"images/bg_images/{args.background_path}"
    num_images = args.image_num
    num_objects = args.object_num 
    colored = args.match_color
    send_hunyuan = args.send_hunyuan.lower() == "y"
    hunyuan_url = args.hunyuan_url
    hunyuan_output_dir = args.hunyuan_output_dir
    all_num_objects = num_images + (num_images // 2)
    alpha_folder = "alpha_images"
    rotate_alpha_folder = r"generated_images/3d"
    # texture_files = [f for f in os.listdir(rotate_alpha_folder) if f.startswith("textured_") and f.endswith(".glb")]
    # print(f"Found {len(texture_files)} textured alpha files in '{rotate_alpha_folder}'")
    # exit()
    # upview vehicle paths
    preprocess_vehicles = [os.path.join("images/upview_vehicles/preprocess", f) for f in os.listdir("images/upview_vehicles/preprocess")]
    postprocess_vehicles = [os.path.join("images/upview_vehicles/postprocess", f) for f in os.listdir("images/upview_vehicles/postprocess")]

    # horizontal vehicle paths
    all_vehicle_paths = [os.path.join("images/vehicles", f) for f in os.listdir("images/vehicles")]
    # all_vehicle_paths = [""]
    
    alpha_paths = []
    shutil.rmtree(img_folder, ignore_errors=True)
    shutil.rmtree(label_folder, ignore_errors=True)
    
    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)
    os.makedirs(alpha_folder, exist_ok=True)
    
    # --PREPARE UPVIEW VEHICLES---
    # for idx, path in enumerate(preprocess_vehicles):
    #     crop_vehicle(path,idx)

    # --LOAD ALPHA IMAGES DIRECTLY (alpha images already prepared)---
    # alpha_paths = [
    #     os.path.join(alpha_folder, f)
    #     for f in os.listdir(alpha_folder)
    #     if f.lower().endswith(".png")
    # ]
    # -- Load ảnh alpha ở alpha_folder đã có texture ở folder 3d
    alpha_paths = [
        os.path.join(alpha_folder, f.replace("textured_", "").replace(".glb", ".png"))
        for f in os.listdir(rotate_alpha_folder)
        if f.lower().endswith(".glb") and f.startswith("textured_")
    ]
    
    print(f"Loaded {len(alpha_paths)} alpha images from '{alpha_folder}'")
    if not alpha_paths:
        print("ERROR: No alpha images found! Please prepare alpha images first.")
        exit(1)
    # img_ori_paths = []
    # checked_paths = set()

    # --- SKIP: select from all_vehicle_paths ---
    # if not alpha_paths:
    #     for i in range(all_num_objects):
    #         while True:
    #             image_path = random.choice(all_vehicle_paths)
    #             if image_path in checked_paths:
    #                 continue
    #             image = cv2.imread(image_path)
    #             if image is not None and check_image_size(image, min_size=100):
    #                 img_ori_paths.append(image_path)
    #                 checked_paths.add(image_path)
    #                 break
    #     print("Chọn được ảnh:", img_ori_paths)
    #     print("Tổng số ảnh đã chọn:", len(img_ori_paths))

    # --- SKIP: remove background & save alpha ---
    # if not alpha_paths:
    #     for img in img_ori_paths:
    #         alpha_path = os.path.join(alpha_folder, os.path.basename(img).replace(".jpg", ".png"))
    #         if not os.path.exists(alpha_path):
    #             obj_img = cv2.imread(img)
    #             rm_background_img = remove_vehicle_background(obj_img)
    #             if rm_background_img is None:
    #                 print(f"No vehicle detected in {img}, skipping...")
    #                 continue
    #             resized_img = minimum_object_size(rm_background_img, max_size=256)[0]
    #             cv2.imwrite(alpha_path, resized_img)
    #             print(f"Saved alpha image: {alpha_path}")
    #         alpha_paths.append(alpha_path)
    
    bg_img = cv2.imread(img_bg_path)
    depth_map = estimate_depth(bg_img)
    depth_map = cv2.resize(depth_map, (bg_img.shape[1], bg_img.shape[0]))
    mask = get_ground_mask(args.background_path)
    box_bg = get_ground_bbox(args.background_path)
    
    
    if mask is not None:
        cv2.imwrite("ground_mask_V1.png", mask*255) 
    
    # Phủ các vết nứt trong mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("ground_mask_V2.png", closed_mask*255)

    # tìm vùng có contour lớn nhất
    contours, _ = cv2.findContours(
        closed_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    largest = max(contours, key=cv2.contourArea)
    ground_mask = np.zeros(mask.shape, dtype=np.uint8)  
    cv2.drawContours(ground_mask, [largest], -1, color=1, thickness=-1)
    cv2.imwrite("ground_mask_final.png", ground_mask*255)

    # Vẽ đa giác bao quanh contour lớn nhất
    pts = np.vstack(largest)
    hull = cv2.convexHull(pts)

    eps = 0.02 * cv2.arcLength(hull, True)
    quad = cv2.approxPolyDP(hull, eps, True)
    ground_polygon = quad.reshape(-1, 2)
    
    # vẽ tam giác và đường vuông góc 
    va_pts_idx, va_angle, edge1, edge2 = find_smallest_angle_point(ground_polygon)
    va_pts = edge1[1]
    triangle_area = np.array([va_pts, edge1[0], edge2[1]], dtype=np.int32)
    base_p1 = np.array(edge1[0], dtype=float)
    base_p2 = np.array(edge2[1], dtype=float)
    midpoint = ((base_p1 + base_p2) / 2).astype(int)
    
    if args.visualize.lower() == "y":
        cv2.line(bg_img, tuple(va_pts.astype(int)), tuple(midpoint), (0,255,255), 3)
        cv2.circle(bg_img, tuple(va_pts.astype(int)), 10, (255,255,255), -1)
        cv2.circle(bg_img, tuple(midpoint), 10, (0,165,255), -1)

    pts_new = np.maximum(ground_polygon - 10, 0)
    if args.visualize.lower() == "y":
        cv2.polylines(bg_img, [triangle_area], isClosed=True, color=(0,0,255), thickness=2)
        cv2.polylines(bg_img, [pts_new], isClosed=True, color=(0,255,0), thickness=2)
        cv2.drawContours(bg_img, [largest], -1, color=(255,0,0), thickness=2)
    cv2.imwrite("ground_mask_visualization.png", bg_img)
    # ---GENERATE IMAGES---
    for i in range(num_images):
        boxes = []
        polygons = []  # Lưu polygon của mỗi object

        # alpha_path_process = alpha_paths[i:i+num_objects] 
        # alpha_path_process = [alpha_paths[0]] * num_objects
        alpha_path_process = random.sample(alpha_paths, min(num_objects, len(alpha_paths))) \
            if len(alpha_paths) >= num_objects \
            else random.choices(alpha_paths, k=num_objects)

        print(f"Processing image {i+1}/{num_images}")
        print("len of alpha paths:", len(alpha_path_process))
        new_bg_img = bg_img.copy()
        visulize_bg = bg_img.copy()
        
        for alpha_path in alpha_path_process:
            if alpha_path == "alpha_images/construction_truck.png":
                continue
             
            alpha_img = cv2.imread(alpha_path, cv2.IMREAD_UNCHANGED)
            alpha_img, bbox = get_tight_bbox_from_alpha(alpha_img)
            alpha_img = cv2.cvtColor(alpha_img, cv2.COLOR_BGRA2RGBA)
            
            """ obstacles paste"""
            new_bg_img, (x1, y1, x2, y2), obj_polygon = paste_object_with_alpha(
                img_bg_path, new_bg_img, alpha_img, alpha_path, ground_polygon, depth_map, 
                ground_mask, colored, boxes=boxes, triangle_pts=triangle_area
            )
            boxes.append([x1, y1, x2, y2])
            polygons.append(obj_polygon)
        
            labeling_custom(img_bg_path, f"{img_folder}/out_{i+1}.png", alpha_path, 
                        x1/new_bg_img.width, y1/new_bg_img.height, 
                        x2/new_bg_img.width, y2/new_bg_img.height)
            
        cv_img = np.array(new_bg_img)
        
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            box_depth = get_box_depth(depth_map, box)
            box_3d_points = create_3d_box_from_mask(
                [x1, y1, x2, y2], box_depth, 
                image=cv_img, 
                height_scale=1.2,
                vanishing_point=va_pts  
            )
            y_rotate_degree = get_rotate_degree(va_pts, midpoint, [(x1, y1), (x2, y2)])
            if args.visualize.lower() == "y":
                cv_img = draw_3d_box(cv_img, box_3d_points, color=(0, 255, 0), thickness=2,
                                    triangle_pts=triangle_area)

            # ===== Log góc giữa P1→P2 và P1→P2' =====
            
            P1 = np.array(box_3d_points[P1_IDX], dtype=float)   # bottom_front_right
            P2 = np.array(box_3d_points[P2_IDX], dtype=float)   # bottom_back_right
            dx_offset, dy_offset = DX_OFFSET, DY_OFFSET
            P2_prime = P1 + np.array([dx_offset, dy_offset], dtype=float)

            v_P1_P2       = P2 - P1
            v_P1_P2_prime = P2_prime - P1

            len_v1 = np.linalg.norm(v_P1_P2)
            len_v2 = np.linalg.norm(v_P1_P2_prime)
            
            x_rotate_degree = get_rotate_degree_cross_line(va_pts, P1, [va_pts, P2_prime])            

            if len_v1 > 1e-6 and len_v2 > 1e-6:
                cos_a = np.dot(v_P1_P2, v_P1_P2_prime) / (len_v1 * len_v2)
                angle_P1P2_P1P2prime = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
                print(f"[ANGLE] P1={tuple(P1.astype(int))}, P2={tuple(P2.astype(int))}, P2'={tuple(P2_prime.astype(int))}")
                print(f"[ANGLE] vec P1→P2       = {v_P1_P2.tolist()}")
                print(f"[ANGLE] vec P1→P2'      = {v_P1_P2_prime.tolist()}")
                print(f"[ANGLE] Góc P1P2 vs P1P2' = {angle_P1P2_P1P2prime:.2f}°")
                print(f"[ANGLE] X ROTATION = {x_rotate_degree:.2f}°")
            else:
                print("[ANGLE] Không tính được góc (vector có độ dài = 0)")

            if send_hunyuan:
                flip_deg = random.choice([0, 180])
                yaw_deg = normalize_yaw(angle_P1P2_P1P2prime + flip_deg) 
                pitch_deg = normalize_yaw(x_rotate_degree)
                alpha_path = alpha_path_process[idx] if idx < len(alpha_path_process) else None
                if alpha_path:
                    result = send_to_hunyuan3d(alpha_path, int(yaw_deg), hunyuan_url, hunyuan_output_dir, int(pitch_deg),turn=1)
                    if result is None:
                        print(f"[SKIP] Hunyuan3D failed for {alpha_path}, skipping rotate paste.")
                        continue

                    rotate_alpha_path = f"{rotate_alpha_folder}/{os.path.basename(alpha_path).replace('.png', '_view.png')}"
                    print(f"Rotate alpha path: {rotate_alpha_path}")

                    if not wait_for_file(rotate_alpha_path, timeout=900, poll_interval=2.0):
                        print(f"[SKIP] Output file not ready: {rotate_alpha_path}, skipping rotate paste.")
                        continue

                    rotate_alpha_img = cv2.imread(rotate_alpha_path, cv2.IMREAD_UNCHANGED)
                    if rotate_alpha_img is None:
                        print(f"[SKIP] Could not read {rotate_alpha_path}, skipping rotate paste.")
                        continue
                    rotate_alpha_img, bbox = get_tight_bbox_from_alpha(rotate_alpha_img)
                    rotate_alpha_img = cv2.cvtColor(rotate_alpha_img, cv2.COLOR_BGRA2RGBA)

                    new_bg_img = Image.fromarray(cv_img)
                    new_bg_img, (r_x1, r_y1, r_x2, r_y2), obj_polygon = paste_object_with_alpha(
                        img_bg_path, new_bg_img, rotate_alpha_img, rotate_alpha_path, ground_polygon, depth_map,
                        ground_mask, colored, boxes=boxes, triangle_pts=triangle_area
                    )
                    boxes.append([r_x1, r_y1, r_x2, r_y2])
                    polygons.append(obj_polygon)

                    labeling_custom(img_bg_path, f"{img_folder}/out_{i+1}.png", rotate_alpha_path,
                                r_x1/new_bg_img.width, r_y1/new_bg_img.height,
                                r_x2/new_bg_img.width, r_y2/new_bg_img.height)

                    box = [r_x1, r_y1, r_x2, r_y2]
                    box_depth = get_box_depth(depth_map, box)
                    box_3d_points = create_3d_box_from_mask(
                        [r_x1, r_y1, r_x2, r_y2], box_depth, 
                        image=cv_img, 
                        height_scale=1.2,
                        vanishing_point=va_pts  
                    )
                    cv_img = np.array(new_bg_img)
        #             cv_img = draw_3d_box(cv_img, box_3d_points, color=(0, 255, 0), thickness=2,
        #             triangle_pts=triangle_area)
        # cv_img = np.array(new_bg_img)

            
        if num_objects % 2 == 1:
            cv2.imwrite(f'{img_folder}/out_{i+1}.png',cv_img)
        else:
            Image.fromarray(cv_img).save(f'{img_folder}/out_{i+1}.png')

    print("Đã tạo thư mục và lưu ảnh vào:", img_folder)
    print("Đã tạo file chứa kết quả paste với alpha blending")