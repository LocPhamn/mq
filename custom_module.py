import numpy as np
import json
import os
import pycocotools.mask as mask_util
import cv2
import shutil
from PIL import Image
from skimage.exposure import match_histograms

# ===== Checking Functions ===== #

def point_in_polygon(x, y, polygon):
    n = len(polygon)
    inside = False

    x0, y0 = polygon[0]
    for i in range(1, n + 1):
        x1, y1 = polygon[i % n]

        if min(y0, y1) < y <= max(y0, y1):
            if x <= max(x0, x1):
                if y0 != y1:
                    x_intersect = (y - y0) * (x1 - x0) / (y1 - y0) + x0
                if x0 == x1 or x <= x_intersect:
                    inside = not inside

        x0, y0 = x1, y1

    return inside

def check_image_size(image, min_size=100):
    h, w = image.shape[:2]
    return h >= min_size and w >= min_size

def check_bottom_line_on_ground(mask, x1, y2, x2):

    if x1 >= x2:
        return False
    bottom_line = mask[y2, x1:x2+1]  
    return np.all(bottom_line == 1)

def check_any_pixel_on_ground(mask, x1,y1, y2, x2):
    h = y2 - y1
    start_y = y1 + int(h * 5/6)
    area = mask[start_y:y2+1, x1:x2+1] 
    return np.any(area == 1)

def check_object_center_perpendicular(x1_obj, y1_obj, x2_obj, y2_obj, x1_ground, x2_ground,y1_ground ,y2_ground):
    obj_center_x = (x1_obj + x2_obj) / 2
    obj_center_y = (y1_obj + y2_obj) / 2
    P = np.array([obj_center_x, obj_center_y])
    ground_start = np.array([x1_ground, y1_ground])
    ground_end = np.array([x2_ground, y2_ground])

    v1 = ground_end - ground_start
    v2 = P - ground_start
    return np.isclose(np.dot(v1, v2), 0, atol=1e-2)

def check_ground_contact(x1_obj, y1_obj, x2_obj, y2_obj, mask):
    h = y2_obj - y1_obj
    obj_third_y = y1_obj + int(h * 3/4)

    area = mask[int(obj_third_y):y2_obj, x1_obj:x2_obj+1]
    return np.all(area == 1)

def check_object_iou_with_each_other(box1,boxes, iou_threshold=0.5):
    """
    Kiểm tra xem box1 có đè lên hoặc bị đè bởi bất kỳ box nào trong danh sách boxes không.
    Sử dụng Intersection over Self (IoS) - tính riêng cho từng box.
    
    - IoS_box1 = intersection / area_box1 (kiểm tra box1 có bị đè không)
    - IoS_box2 = intersection / area_box2 (kiểm tra box2 có bị đè không)
    
    Returns:
        True nếu box1 KHÔNG đè lên và KHÔNG bị đè bởi box nào
        False nếu có overlap vượt ngưỡng (đè hoặc bị đè)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    if not boxes:
        print(f"[DEBUG] List is empty - returning True immediately")
        return True 
    
    for x1_2, y1_2, x2_2, y2_2 in boxes:
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        if inter_area == 0:
            continue  # Không có overlap, kiểm tra box tiếp theo
            
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

        ios_box1 = inter_area / box1_area if box1_area > 0 else 0  # box1 bị đè bởi box2
        ios_box2 = inter_area / box2_area if box2_area > 0 else 0  # box2 bị đè bởi box1
        
        if ios_box1 > iou_threshold or ios_box2 > iou_threshold:
            if ios_box1 > iou_threshold:
                print(f"[DEBUG] box1 bị đè: IoS_box1={ios_box1:.3f} > {iou_threshold:.3f}")
            if ios_box2 > iou_threshold:
                print(f"[DEBUG] box2 bị đè: IoS_box2={ios_box2:.3f} > {iou_threshold:.3f}")
            return False
            
    return True

def check_overlap_boxes(box1, x1s, x2s,y1s,y2s):
    x1_1, y1_1, x2_1, y2_1 = box1
    
    if not x1s or not x2s or not y1s or not y2s:
        return True 
    
    for x1_2, y1_2, x2_2, y2_2 in zip(x1s, y1s, x2s, y2s):
        if x1_1 > x2_2 or x2_1 < x1_2 and y1_1 > y2_2 or y2_1 < y1_2:
            return True

    return False

def check_object_in_box(box1, box2, iou_threshold=0.8):
    """
    Kiểm tra IoU giữa box1 và box2.
    box format: (x1, y1, x2, y2) không bắt buộc x1<x2 hay y1<y2 (hàm sẽ chuẩn hóa).
    Trả về:
      - nếu return_iou False: bool (True nếu IoU >= iou_threshold)
      - nếu return_iou True: (bool, iou)
    """
    # Unpack
    x1_obj, y1_obj, x2_obj, y2_obj = box1
    x1_box, y1_box, x2_box, y2_box = box2

    # Chuẩn hóa để đảm bảo x1<=x2, y1<=y2
    xa1, xa2 = min(x1_obj, x2_obj), max(x1_obj, x2_obj)
    ya1, ya2 = min(y1_obj, y2_obj), max(y1_obj, y2_obj)
    xb1, xb2 = min(x1_box, x2_box), max(x1_box, x2_box)
    yb1, yb2 = min(y1_box, y2_box), max(y1_box, y2_box)

    # Giao
    xi1 = max(xa1, xb1)
    yi1 = max(ya1, yb1)
    xi2 = min(xa2, xb2)
    yi2 = min(ya2, yb2)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)

    union_area = area_a + area_b - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0
    is_match = iou >= iou_threshold
    print(f"iou_area: {iou}, is_match: {is_match}")
    return is_match

def check_polygon_area(label_path ,log_path, crop_offset, original_shape, crop_shape, threshold_ratio=0.05):
    
    x_offset, y_offset = crop_offset
    orig_h, orig_w = original_shape[:2]
    crop_h, crop_w = crop_shape[:2]
    crop_area = crop_h * crop_w
    
    with open(label_path, "r") as f:
        lines = f.readlines()
        
    if not lines:  # file rỗng
        print(f"Warning: Empty label file {label_path}")

    polygon_data = []
    with open(log_path, "a") as log_file:
            log_file.write(
                f"{label_path}:\n "
            )
    for line in lines:
        parts = list(map(float, line.strip().split()))
        class_id = parts[0]
        coords = list(map(float, parts[1:]))
        
        if len(coords) < 6:  
            print(f"Warning: Polygon with less than 3 points in {label_path}")
            
        points = []
        for i in range(0, len(coords), 2):
            x_norm, y_norm = coords[i], coords[i+1]
            x_pixel = x_norm * orig_w
            y_pixel = y_norm * orig_h
            points.append([x_pixel, y_pixel])
          
        crop_points = []
        for point in points:
            x_crop = point[0] - x_offset
            y_crop = point[1] - y_offset
            crop_points.append([x_crop, y_crop])
        
        points = np.array(points, dtype=float)  
        area = 0.0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]

        area = abs(area) / 2.0
        polygon_data.append((line,area, class_id))
        
        with open(log_path, "a") as log_file:
            log_file.write(
                f"ID{class_id}: Area={area / crop_area}\n"
            )
     # **Bước 2: Tìm object có diện tích lớn nhất**
    max_area = max(data[1] for data in polygon_data)
    
    valid_lines = []
    removed_count = 0
    for line, area, class_id in polygon_data:
        area_ratio = area / max_area  
        crop_ratio = area / crop_area 
        
        if area_ratio < threshold_ratio:
            ox = get_bbox_from_points(points)
            removed_count += 1
        else:
            print(f"✅ Keeping class {class_id}: area={area:.2f}, ratio={area_ratio:.3f} >= {threshold_ratio}")
            with open(log_path, "a") as log_file:
                log_file.write(f" → KEPT\n")
            valid_lines.append(line)
    
    # **Bước 4: Ghi lại file với chỉ các dòng hợp lệ**
    with open(label_path, "w") as f:
        f.writelines(valid_lines)

def check_other_polygon_area(other_path, removed_object_boxs, crop_offset, original_shape, crop_shape):
    
    x_offset, y_offset = crop_offset
    orig_h, orig_w = original_shape[:2]
    crop_h, crop_w = crop_shape[:2]
    crop_area = crop_h * crop_w
    
    with open(other_path, "r") as f:
        lines = f.readlines()
        
    if not lines:  # file rỗng
        print(f"Warning: Empty label file {other_path}")
    
    valid_other_lines = []
    
    for other_line in lines:
        other_parts = list(map(float, other_line.strip().split()))
        other_class_id = other_parts[0]
        other_coords = list(map(float, other_parts[1:]))
        if len(other_coords) < 6:  
            print(f"Warning: Polygon with less than 3 points in {other_path}")
                
        # Convert normalized coords to pixel coords (ảnh gốc)
        other_points = []
        for i in range(0, len(other_coords), 2):
            x_norm, y_norm = other_coords[i], other_coords[i+1]
            x_pixel = x_norm * orig_w
            y_pixel = y_norm * orig_h
            other_points.append([x_pixel, y_pixel])
        
        crop_points = []
        for point in other_points:
            x_crop = point[0] - x_offset
            y_crop = point[1] - y_offset
            crop_points.append([x_crop, y_crop])
        
        other_points = np.array(crop_points, dtype=float)  
        other_box = get_bbox_from_points(other_points)
                
        inside_object = False
        for removed_box in removed_object_boxs:
    
            if check_object_in_box(other_box, removed_box, iou_threshold=0.5):
                print(f"❌ remove {other_class_id} in path {other_path}")
                inside_object = True
                break
        
        if inside_object:
            continue  # Bỏ qua dòng này
        else:
            valid_other_lines.append(other_line)
            print(f"✅ keep {other_class_id} in path {other_path}")
                
    with open(other_path, "w") as f:
        f.writelines(valid_other_lines)        
# ===== Get Label Stuff ====== #

def get_bbox_from_points(points):
    if len(points) == 0:
        return None
    
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)
    
    return xmin, ymin, xmax, ymax

def get_ground_bbox(bg_path):
    bg_path = bg_path.split(".")[0]

    json_file = f"outputs/grounded_sam2_local_demo/{bg_path}_results.json"
    
    if os.path.exists(json_file):
        print("JSON file found.") 
        
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    annotations = data['annotations']
    for ann in annotations:
        if ann['class_name'] == 'ground':
            x1, y1, x2, y2 = map(int, ann['bbox'])
            box = [x1, y1, x2, y2]
            print(f"Ground bounding box: {box}")
            return ann['bbox']
    return None

def get_ground_mask(bg_path):
    bg_path = bg_path.split(".")[0]
    json_file = f"outputs/grounded_sam2_local_demo/{bg_path}_results.json"
    
    if os.path.exists(json_file):
        print("JSON file found.") 
        
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    annotations = data['annotations']
    img_height = data['img_height']
    img_width = data['img_width']
    
    for ann in annotations:
        if ann['class_name'] == 'ground':
            # Lấy RLE segmentation
            rle = ann['segmentation']
            
            # Decode RLE thành binary mask
            mask = mask_util.decode(rle)
            
            print(f"Ground mask shape: {mask.shape}")
            print(f"Mask data type: {mask.dtype}")
            print(f"Unique values in mask: {np.unique(mask)}")
            
            return mask

def get_tight_bbox_from_alpha(rgba_img):
    """
    Tính bounding box chính xác từ alpha channel và crop ảnh
    
    Args:
        rgba_img: ảnh RGBA với alpha channel
    
    Returns:
        cropped_img: ảnh đã crop theo object
        bbox: [x1, y1, x2, y2] - bounding box mới (relative to cropped image, luôn là [0, 0, w, h])
    """
    if rgba_img.shape[2] != 4:
        print("Warning: Image doesn't have alpha channel")
        return rgba_img, [0, 0, rgba_img.shape[1], rgba_img.shape[0]]
    
    alpha_channel = rgba_img[:, :, 3]
    
    # Tìm các pixel có alpha > 0
    rows = np.any(alpha_channel > 0, axis=1)
    cols = np.any(alpha_channel > 0, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        print("Warning: No non-transparent pixels found")
        return rgba_img, [0, 0, rgba_img.shape[1], rgba_img.shape[0]]
    
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    
    # Crop ảnh theo bounding box
    cropped_img = rgba_img[y1:y2+1, x1:x2+1]
    
    # Box mới relative to cropped image
    new_h, new_w = cropped_img.shape[:2]
    bbox = [0, 0, new_w, new_h]
    
    print(f"Original size: {rgba_img.shape[:2]}, Cropped to: {cropped_img.shape[:2]}")
    print(f"Tight bbox: [{x1}, {y1}, {x2}, {y2}]")
    
    return cropped_img, bbox

def yolo_seg_to_mask(image_path, label_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    mask = np.zeros((h, w), dtype=np.uint8)
    true_pts = []
    with open(label_path, "r") as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            cls = int(parts[0])
            points = np.array(parts[1:]).reshape(-1, 2)
            
            pts = []
            for x, y in points:
                pts.append([int(x * w), int(y * h)])
            pts = np.array(pts, np.int32)
            true_pts.append(pts)
            cv2.fillPoly(mask, [pts], 255)

    mask = mask.astype(np.float32) / 255.0
    return mask, true_pts

def get_labels_from_yolo_seg(label_path):
    labels = []
    lines = 0
    with open(label_path, "r") as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            cls = int(parts[0])
            points = np.array(parts[1:]).reshape(-1, 2)
            labels.append((cls, points))
    print(f"Total labels read from : {lines}")
    return labels

def label_process(image_path, label_path, matrix, image):
    """
    Chuyển đổi label segmentation từ ảnh gốc sang ảnh sau perspective transformation
    
    Args:
        image_path: đường dẫn ảnh gốc
        label_path: đường dẫn file label YOLO segmentation
        matrix: ma trận perspective transform H
        image: ảnh sau khi transform (để lấy kích thước mới)
    
    Returns:
        new_labels: list các polygon đã được transform theo định dạng YOLO
    """
    # Đọc ảnh gốc để lấy kích thước
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Cannot read image {image_path}")
        return []
    
    orig_h, orig_w = img.shape[:2]
    
    # Lấy kích thước ảnh mới sau transform
    new_h, new_w = image.shape[:2]
    
    # Đọc file label
    if not os.path.exists(label_path):
        print(f"ERROR: Label file not found: {label_path}")
        return []
    
    new_labels = []
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    print("====================================")
    print(f"Number of lines in label file: {len(lines)}")
    for line in lines:
        parts = line.strip().split()
        print(parts)
        if len(parts) < 3:  # class_id + ít nhất 1 cặp tọa độ
            continue
        
        class_id = int(parts[0])
        
        # Parse normalized coordinates (x1, y1, x2, y2, ...)
        coords = list(map(float, parts[1:]))
        
        # Chuyển từ normalized (0-1) về pixel coordinates của ảnh gốc
        points = []
        for i in range(0, len(coords), 2):
            x_norm = coords[i]
            y_norm = coords[i + 1]
            x_pixel = x_norm * orig_w
            y_pixel = y_norm * orig_h
            points.append([x_pixel, y_pixel])
        
        points = np.array(points, dtype=np.float32)
        
        # Transform points bằng perspective matrix
        # cv2.perspectiveTransform yêu cầu shape (1, N, 2)
        points_reshaped = points.reshape(1, -1, 2)
        transformed_points = cv2.perspectiveTransform(points_reshaped, matrix)
        transformed_points = transformed_points.reshape(-1, 2)
        
        # Clip points về trong boundaries của ảnh mới
        transformed_points[:, 0] = np.clip(transformed_points[:, 0], 0, new_w)
        transformed_points[:, 1] = np.clip(transformed_points[:, 1], 0, new_h)
        
        # Normalize lại về (0-1) theo kích thước ảnh mới
        normalized_points = []
        for pt in transformed_points:
            x_norm_new = pt[0] / new_w
            y_norm_new = pt[1] / new_h
            normalized_points.extend([x_norm_new, y_norm_new])
        
        # Tạo label mới: [class_id, x1, y1, x2, y2, ...]
        new_label = [class_id] + normalized_points
        new_labels.append(new_label)
    
    return new_labels


def copy_object(img_path, box):
    img_ori = cv2.imread(img_path)
    h, w, c = img_ori.shape
    
    x_center, y_center, bw, bh = box
    
    x1 = int((x_center - bw / 2) * w)
    x2 = int((x_center + bw / 2) * w)
    y1 = int((y_center - bh / 2) * h)
    y2 = int((y_center + bh / 2) * h)

    obj_img = img_ori[y1:y2, x1:x2]
    return obj_img

## ===== Color Stuff ===== ##
def match_color(foreground_big, bg_img):
    foreground_array = np.array(foreground_big.convert("RGBA"))
    bg_array = np.array(bg_img.convert("RGBA"))
    
    # Match histogram toàn bộ
    foreground_matched = match_histograms(
        foreground_array[:, :, :3],  # Chỉ RGB
        bg_array[:, :, :3],
        channel_axis=-1
    )
    
    # Tạo RGBA mới, giữ nguyên alpha gốc
    result = np.zeros_like(foreground_array)
    result[:, :, :3] = foreground_matched
    result[:, :, 3] = foreground_array[:, :, 3]  # Giữ nguyên alpha
    
    # Chỉ áp dụng color matching cho vùng có alpha > 0
    alpha_mask = foreground_array[:, :, 3] == 0
    result[alpha_mask] = [0, 0, 0, 0]  # Set transparent pixels về 0
    
    foreground_big_colored = Image.fromarray(result, 'RGBA')
    enhance_contrastd = enhance_contrast(foreground_big_colored, foreground_big)
    return enhance_contrastd

def enhance_contrast(image, ref_image, dark_factor=0.8):
    image_array = np.array(image)        
    ref_image_array = np.array(ref_image) 
    
    # Channels đã đúng: R=0, G=1, B=2, A=3
    r, g, b, a = ref_image_array[:,:,0], ref_image_array[:,:,1], ref_image_array[:,:,2], ref_image_array[:,:,3]
    
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    
    threhold1 = 60
    threhold2 = 230
    
    dark_mask = luminance < threhold1
    light_mask = luminance > threhold2
    midtone_mask = (luminance >= threhold1) & (luminance <= threhold2)
    
    # Xử lý 3 channels RGB (giữ nguyên alpha channel)
    for i, channel in enumerate([image_array[:,:,0], image_array[:,:,1], image_array[:,:,2]]):
        channel[dark_mask] = ref_image_array[:,:,i][dark_mask]
        channel[light_mask] = ref_image_array[:,:,i][light_mask]
        channel[midtone_mask] = np.clip(channel[midtone_mask] * dark_factor, threhold1, threhold2)
    
    # Không cần convert color, trả về PIL Image trực tiếp
    enhance_image = Image.fromarray(image_array, 'RGBA')
    return enhance_image

def rgba_to_gray_with_alpha(rgba_img):
    """
    Chuyển ảnh RGBA sang ảnh grayscale, giữ nguyên kênh Alpha.
    Input:
        rgba_img: ảnh numpy (H, W, 4)
    Output:
        gray_rgba: ảnh numpy (H, W, 4) gồm [gray, gray, gray, alpha]
    """
    if rgba_img.shape[2] != 4:
        raise ValueError("Ảnh phải có 4 kênh (RGBA)")

    # Tách kênh
    rgb = rgba_img[:, :, :3]
    alpha = rgba_img[:, :, 3]

    # Chuyển RGB → Grayscale
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Ghép lại thành RGBA (3 kênh grayscale + alpha gốc)
    gray_rgb = cv2.merge([gray, gray, gray, alpha])

    return gray_rgb

def get_mean_histogram(hist, top_idx):
    sum_hist = []
    sum_pixel = []
    
    for i in top_idx:
        sum_hist.append(hist[i] * i)
        sum_pixel.append(hist[i])
    mean_hist = sum(sum_hist) / sum(sum_pixel)
    return mean_hist

def lightness_matching(obj_img,bg_img,paste_start_y,paste_end_y,paste_start_x,paste_end_x,top_idx=5):
    """
    obj_img: PIL Image hoặc numpy array với 4 kênh (RGBA)
    bg_img: PIL Image hoặc numpy array (RGB hoặc RGBA) - will be resized to obj size if needed
    top_idx: số bins hàng đầu để tính mean histogram
    Trả về: PIL Image (RGBA) với lightness (V) đã được điều chỉnh
    """
    # Convert to numpy
    obj_arr = np.array(obj_img)
    bg_arr = np.array(bg_img)
    
    bg_h, bg_w = bg_arr.shape[:2]
    paste_start_y = int(max(0, min(paste_start_y, bg_h)))
    paste_end_y = int(max(0, min(paste_end_y, bg_h)))
    paste_start_x = int(max(0, min(paste_start_x, bg_w)))
    paste_end_x = int(max(0, min(paste_end_x, bg_w)))
    
    bg_arr = bg_arr[paste_start_y:paste_end_y, paste_start_x:paste_end_x]
  
    alpha = obj_arr[:, :, 3]

    # Use only RGB channels for HSV conversion
    obj_rgb = obj_arr[:, :, :3].astype(np.uint8)
    bg_rgb = bg_arr[:, :, :3].astype(np.uint8)

    # Convert to HSV (input assumed RGB)
    obj_hsv = cv2.cvtColor(obj_rgb, cv2.COLOR_RGB2HSV)
    bg_hsv = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2HSV)

    obj_v = obj_hsv[:, :, 2]
    bg_v = bg_hsv[:, :, 2]

    # Build mask of object pixels (alpha > 0)
    obj_mask = (alpha > 0).astype(np.uint8)
    # bg_mask = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2GRAY)

    if obj_mask.sum() == 0:
        return Image.fromarray(obj_arr, 'RGBA')

    obj_hist = cv2.calcHist([obj_v], [0], obj_mask, [256], [0, 256])
    bg_hist = cv2.calcHist([bg_v], [0], None, [256], [0, 256])

    obj_hist_list = [float(i[0]) for i in obj_hist]
    bg_hist_list = [float(i[0]) for i in bg_hist]

    top_obj_idx = np.argsort(obj_hist_list)[-top_idx:][::-1]
    top_bg_idx = np.argsort(bg_hist_list)[-top_idx:][::-1]

    obj_mean_hist = get_mean_histogram(obj_hist_list, top_obj_idx)
    bg_mean_hist = get_mean_histogram(bg_hist_list, top_bg_idx)

    value_change = float(bg_mean_hist - 1 * obj_mean_hist)
    
    print(f"Object mean V: {obj_mean_hist}, Background mean V: {bg_mean_hist}, Value change: {value_change}")

    new_hsv = obj_hsv.copy().astype(np.float32)
    new_v = new_hsv[:, :, 2]
    new_v = np.clip(new_v + value_change, 0, 255)
    new_v[obj_mask == 0] = obj_hsv[:, :, 2][obj_mask == 0]
    new_hsv[:, :, 2] = new_v

    # Back to uint8 and convert to RGB
    new_hsv = new_hsv.astype(np.uint8)
    new_rgb = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2RGB)

    # Reattach alpha and return PIL Image RGBA
    out_arr = np.dstack([new_rgb, alpha]).astype(np.uint8)
    out_img = Image.fromarray(out_arr, 'RGBA')
    return out_img

def get_mean_color(image,box):
    pad = 40  # mở rộng vùng lấy mẫu
    x1, y1, x2, y2 = box
    roi = image[
        max(0, y1-pad):min(image.shape[0], y2+pad),
        max(0, x1-pad):min(image.shape[1], x2+pad)
    ]

    mean_bg_color = roi.astype(np.float32).mean(axis=(0, 1))
    return mean_bg_color
    
def color_transfer_blend(obj_img, bg_img, mean_bg_color, alpha=0.3):
    """
    Blend mean background color lên object với tỷ lệ alpha
    alpha: 0.0-1.0, càng cao càng gần màu background
    """
    obj_array = np.array(obj_img).astype(np.float32)
    
    # Tạo overlay layer với màu trung bình
    overlay = np.ones_like(obj_array[:,:,:3]) * mean_bg_color
    
    # Blend theo alpha, chỉ áp dụng cho RGB channels
    obj_array[:,:,:3] = (1 - alpha) * obj_array[:,:,:3] + alpha * overlay
    
    obj_array = np.clip(obj_array, 0, 255).astype(np.uint8)
    return Image.fromarray(obj_array)


# ===== Shape Stuff ====== #
def project_point_to_line(point, line_point1, line_point2):
    """
    Hạ vuông góc từ điểm xuống đường thẳng (projection)
    
    Args:
        point: điểm cần hạ vuông góc [x, y]
        line_point1: điểm đầu của đường thẳng [x, y]
        line_point2: điểm cuối của đường thẳng [x, y]
    
    Returns:
        projected_point: điểm vuông góc trên đường thẳng [x, y]
    """
    px, py = point
    x1, y1 = line_point1
    x2, y2 = line_point2
    
    # Vector của đường thẳng
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        return line_point1
    
    # Tính tham số t cho projection
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    
    # Tính điểm projection
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    
    return [int(proj_x), int(proj_y)]

def get_object_mask_from_region(image, bbox):
    """
    Lấy mask của object từ vùng bbox trên ảnh
    
    Args:
        image: ảnh đã paste object (numpy array hoặc PIL Image)
        bbox: [x1, y1, x2, y2] - bounding box
    
    Returns:
        mask: binary mask của object trong vùng bbox
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    x1, y1, x2, y2 = [int(x) for x in bbox]
    
    # Crop vùng bbox
    region = image[y1:y2, x1:x2]
    
    # Nếu là ảnh có alpha channel
    if region.shape[2] == 4:
        mask = region[:, :, 3] > 0
    else:
        # Chuyển sang grayscale và threshold
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask = mask > 0
    
    return mask.astype(np.uint8)

def create_3d_box_from_mask(bbox_2d, depth_value, image=None, height_scale=1.2, vanishing_point=None):
    """
    Tạo box 3D từ bounding box 2D, depth value, mask và vanishing point
    
    Args:
        bbox_2d: [x1, y1, x2, y2] - bounding box 2D
        depth_value: giá trị depth (0-1, càng lớn càng xa)
        image: ảnh chứa object (để extract mask), nếu None thì dùng method cũ
        height_scale: scale cho chiều cao của box 3D (mặc định 1.2)
        vanishing_point: điểm tụ [x, y] để tính perspective chính xác
    
    Returns:
        box_3d_points: 8 điểm của box 3D
    """
    x1, y1, x2, y2 = [int(x) for x in bbox_2d]
    width = x2 - x1
    height = y2 - y1
    
    # Tính offset theo depth (càng xa thì offset càng nhỏ do perspective)
    depth_scale = 1 - (depth_value * 0.4)  # Scale từ 1.0 đến 0.6
    offset_ratio = 0.3 * depth_scale  # Tăng offset ratio để rõ hơn
    
    if image is not None:
        # Lấy mask của object
        try:
            mask = get_object_mask_from_region(image, bbox_2d)
            
            if mask.size > 0 and np.any(mask > 0):
                # Tìm bottom line thực tế của object (hàng có pixel cuối cùng)
                rows_with_pixels = np.where(np.any(mask > 0, axis=1))[0]
                if len(rows_with_pixels) > 0:
                    bottom_row = rows_with_pixels[-1]  # Hàng cuối cùng có pixel
                    top_row = rows_with_pixels[0]      # Hàng đầu tiên có pixel
                    
                    # Tìm left và right columns thực tế ở bottom
                    bottom_line = mask[bottom_row, :]
                    cols_at_bottom = np.where(bottom_line > 0)[0]
                    
                    if len(cols_at_bottom) > 0:
                        left_col = cols_at_bottom[0]
                        right_col = cols_at_bottom[-1]
                        
                        # Tính tọa độ thực tế trong ảnh gốc
                        actual_bottom_y = y1 + bottom_row
                        actual_top_y = y1 + top_row
                        actual_left_x = x1 + left_col
                        actual_right_x = x1 + right_col
                        
                        # 4 điểm mặt trước (front face)
                        bottom_front_left = [actual_left_x, actual_bottom_y]
                        bottom_front_right = [actual_right_x, actual_bottom_y]
                        top_front_left = [actual_left_x, actual_top_y]
                        top_front_right = [actual_right_x, actual_top_y]
                        
                        # Nếu có vanishing point, sử dụng perspective chính xác
                        if vanishing_point is not None:
                            vp_x, vp_y = vanishing_point
                            
                            # Vẽ đường thẳng từ vanishing point tới 4 điểm front
                            # Tính điểm mặt sau bằng cách di chuyển về phía vanishing point
                            
                            # Tính khoảng cách từ vanishing point đến các điểm front
                            # Mặt sau sẽ gần vanishing point hơn (theo depth)
                            back_ratio = offset_ratio
                            
                            # Bottom back left: di chuyển từ bottom_front_left về phía VP
                            dx_bl = vp_x - bottom_front_left[0]
                            dy_bl = vp_y - bottom_front_left[1]
                            bottom_back_left = [
                                int(bottom_front_left[0] + dx_bl * back_ratio),
                                int(bottom_front_left[1] + dy_bl * back_ratio)
                            ]
                            
                            # Bottom back right
                            dx_br = vp_x - bottom_front_right[0]
                            dy_br = vp_y - bottom_front_right[1]
                            bottom_back_right = [
                                int(bottom_front_right[0] + dx_br * back_ratio),
                                int(bottom_front_right[1] + dy_br * back_ratio)
                            ]
                            
                            # Top back left
                            dx_tl = vp_x - top_front_left[0]
                            dy_tl = vp_y - top_front_left[1]
                            top_back_left = [
                                int(top_front_left[0] + dx_tl * back_ratio),
                                int(top_front_left[1] + dy_tl * back_ratio)
                            ]
                            
                            # Top back right
                            dx_tr = vp_x - top_front_right[0]
                            dy_tr = vp_y - top_front_right[1]
                            top_back_right = [
                                int(top_front_right[0] + dx_tr * back_ratio),
                                int(top_front_right[1] + dy_tr * back_ratio)
                            ]
                        else:
                            # Fallback: sử dụng offset đơn giản
                            x_offset = width * offset_ratio
                            y_offset = height * offset_ratio
                            
                            bottom_back_right = [actual_right_x - x_offset, actual_bottom_y - y_offset]
                            bottom_back_left = [actual_left_x - x_offset, actual_bottom_y - y_offset]
                            top_back_right = [actual_right_x - x_offset, actual_top_y - y_offset]
                            top_back_left = [actual_left_x - x_offset, actual_top_y - y_offset]
                        
                        box_3d_points = np.array([
                            bottom_front_left, bottom_front_right, bottom_back_right, bottom_back_left,
                            top_front_left, top_front_right, top_back_right, top_back_left
                        ], dtype=np.int32)
                        
                        return box_3d_points
        except Exception as e:
            print(f"Warning: Could not extract mask from object, using bbox method. Error: {e}")
    
    # Fallback: sử dụng bbox đơn giản nếu không có mask
    bottom_front_left = [x1, y2]
    bottom_front_right = [x2, y2]
    top_front_left = [x1, y1]
    top_front_right = [x2, y1]
    
    if vanishing_point is not None:
        vp_x, vp_y = vanishing_point
        back_ratio = offset_ratio
        
        dx_bl = vp_x - x1
        dy_bl = vp_y - y2
        bottom_back_left = [int(x1 + dx_bl * back_ratio), int(y2 + dy_bl * back_ratio)]
        
        dx_br = vp_x - x2
        dy_br = vp_y - y2
        bottom_back_right = [int(x2 + dx_br * back_ratio), int(y2 + dy_br * back_ratio)]
        
        dx_tl = vp_x - x1
        dy_tl = vp_y - y1
        top_back_left = [int(x1 + dx_tl * back_ratio), int(y1 + dy_tl * back_ratio)]
        
        dx_tr = vp_x - x2
        dy_tr = vp_y - y1
        top_back_right = [int(x2 + dx_tr * back_ratio), int(y1 + dy_tr * back_ratio)]
    else:
        x_offset = width * offset_ratio
        y_offset = height * offset_ratio
        bottom_back_right = [x2 - x_offset, y2 - y_offset]
        bottom_back_left = [x1 - x_offset, y2 - y_offset]
        top_back_right = [x2 - x_offset, y1 - y_offset]
        top_back_left = [x1 - x_offset, y1 - y_offset]
    
    box_3d_points = np.array([
        bottom_front_left, bottom_front_right, bottom_back_right, bottom_back_left,
        top_front_left, top_front_right, top_back_right, top_back_left
    ], dtype=np.int32)
    
    return box_3d_points

def draw_3d_box(image, box_3d_points, color=(0, 255, 0), thickness=2):
    """
    Vẽ box 3D lên ảnh
    
    Args:
        image: ảnh để vẽ lên
        box_3d_points: 8 điểm của box 3D
        color: màu của box (B, G, R)
        thickness: độ dày của đường vẽ
    
    Returns:
        image: ảnh đã vẽ box 3D
    """
    img = image.copy()
    
    # Vẽ 4 cạnh mặt bottom
    cv2.line(img, tuple(box_3d_points[0]), tuple(box_3d_points[1]), (255,0,0), thickness)
    cv2.line(img, tuple(box_3d_points[1]), tuple(box_3d_points[2]), (255,0,0), thickness)
    cv2.line(img, tuple(box_3d_points[2]), tuple(box_3d_points[3]), (255,0,0), thickness)
    cv2.line(img, tuple(box_3d_points[3]), tuple(box_3d_points[0]), (255,0,0), thickness)
    
    # Vẽ tên point
    cv2.putText(img, 'P0', tuple(box_3d_points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(img, 'P1', tuple(box_3d_points[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(img, 'P2', tuple(box_3d_points[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(img, 'P3', tuple(box_3d_points[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    # Vẽ 4 cạnh mặt top
    cv2.line(img, tuple(box_3d_points[4]), tuple(box_3d_points[5]), (0,0,255), thickness)
    cv2.line(img, tuple(box_3d_points[5]), tuple(box_3d_points[6]), (0,0,255), thickness)
    cv2.line(img, tuple(box_3d_points[6]), tuple(box_3d_points[7]), (0,0,255), thickness)
    cv2.line(img, tuple(box_3d_points[7]), tuple(box_3d_points[4]), (0,0,255), thickness)
    
    # Vẽ tên point
    cv2.putText(img, 'P4', tuple(box_3d_points[4]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(img, 'P5', tuple(box_3d_points[5]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(img, 'P6', tuple(box_3d_points[6]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(img, 'P7', tuple(box_3d_points[7]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    # Vẽ 4 cạnh dọc nối bottom và top
    cv2.line(img, tuple(box_3d_points[0]), tuple(box_3d_points[4]), color, thickness)
    cv2.line(img, tuple(box_3d_points[1]), tuple(box_3d_points[5]), color, thickness)
    cv2.line(img, tuple(box_3d_points[2]), tuple(box_3d_points[6]), color, thickness)
    cv2.line(img, tuple(box_3d_points[3]), tuple(box_3d_points[7]), color, thickness)
    
    return img

def get_box_depth(depth_map, bbox_2d):
    """
    Tính depth trung bình của vùng bounding box
    
    Args:
        depth_map: depth map của ảnh
        bbox_2d: [x1, y1, x2, y2] - bounding box 2D
    
    Returns:
        mean_depth: giá trị depth trung bình (0-1)
    """
    x1, y1, x2, y2 = [int(x) for x in bbox_2d]
    depth_region = depth_map[y1:y2, x1:x2]
    mean_depth = np.mean(depth_region)
    return mean_depth

if __name__ == "__main__":
    path = "images/stockbridge_bg/0b51634f2a2df273ab3c_jpg.rf.44a760f8e5b8b39f09bb632283e13451.jpg"
    base_name = os.path.basename(path).split(".jpg")[0]
    label = f"labels/bg_images/labels/{base_name}.txt"
    
    yolo_seg_to_mask(path, label, "ground_mask.png")
    
    

