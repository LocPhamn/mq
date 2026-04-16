import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import argparse

parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("--image_path","-i",type=str, default="bg (6).png", help="Path to the input image")
parser.add_argument("--image_root", type=str, default="images/bg_images", help="Root folder when --image_path is just a file name")
parser.add_argument("--predict_repeats", type=int, default=1, help="Number of repeated GDINO predicts per image")

agrs = parser.parse_args()

"""
Hyper parameters
"""

TEXT_PROMPT = "yellow wall. ground. road. asphalt. pavement. sidewalk. dirt. soil. gravel. stone ground. floor. drivable surface."

GROUND_SYNONYMS = (
    "ground",
    "road",
    "asphalt",
    "pavement",
    "sidewalk",
    "dirt",
    "soil",
    "gravel",
    "stone",
    "floor",
    "drivable",
)


def resolve_image_path(image_path_arg, image_root):
    p = Path(image_path_arg)
    if p.exists():
        return str(p)
    candidate = Path(image_root) / image_path_arg
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(f"Image not found: {image_path_arg} (also checked {candidate})")


IMG_PATH = resolve_image_path(agrs.image_path, agrs.image_root)
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swinb_cogcoor.pth"
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs/grounded_sam2_local_demo")
DUMP_JSON_RESULTS = True

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# environment settings
# use bfloat16

# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("DEVICE:", DEVICE)
# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = TEXT_PROMPT
img_path = IMG_PATH
image_source, image = load_image(img_path)

sam2_predictor.set_image(image_source)
boxes, confidences, labels = None, None, None
for _ in range(max(1, int(agrs.predict_repeats))):
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE
    )

# process the box prompt for SAM 2
h, w, _ = image_source.shape
boxes = boxes * torch.Tensor([w, h, w, h])
input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()


# FIXME: figure how does this influence the G-DINO model
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

masks, scores, logits = sam2_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)

"""
Post-process the output of the model to get the masks, scores, and logits for visualization
"""
# convert the shape to (n, H, W)
if masks.ndim == 4:
    masks = masks.squeeze(1)

masks = masks.astype(bool)
confidences = confidences.detach().cpu().numpy().astype(float).tolist()
raw_class_names = [str(x) for x in labels]


def normalize_label(name):
    label = name.lower().strip().replace("_", " ").replace("-", " ").rstrip(".")
    if any(keyword in label for keyword in GROUND_SYNONYMS):
        return "ground"
    return label


normalized_class_names = [normalize_label(name) for name in raw_class_names]
ground_indices = [i for i, name in enumerate(normalized_class_names) if name == "ground"]

has_ground_mask = False
if len(ground_indices) > 0 and len(masks) > 0:
    merged_ground = np.zeros((h, w), dtype=bool)
    for idx in ground_indices:
        merged_ground |= masks[idx]

    if merged_ground.any():
        gy, gx = np.where(merged_ground)
        merged_ground_box = np.array(
            [
                float(gx.min()),
                float(gy.min()),
                float(gx.max()),
                float(gy.max()),
            ],
            dtype=np.float32,
        )
        merged_ground_score = float(max(confidences[idx] for idx in ground_indices))

        keep_indices = [i for i in range(len(normalized_class_names)) if i not in ground_indices]
        merged_masks = [masks[i] for i in keep_indices]
        merged_boxes = [input_boxes[i] for i in keep_indices]
        merged_scores = [float(confidences[i]) for i in keep_indices]
        merged_names = [normalized_class_names[i] for i in keep_indices]

        merged_masks.append(merged_ground)
        merged_boxes.append(merged_ground_box)
        merged_scores.append(merged_ground_score)
        merged_names.append("ground")

        masks = np.stack(merged_masks, axis=0)
        input_boxes = np.stack(merged_boxes, axis=0).astype(np.float32)
        confidences = merged_scores
        class_names = merged_names
        has_ground_mask = True
    else:
        print("[WARN] Ground-like labels found but merged ground mask is empty.")
        class_names = normalized_class_names
else:
    print("[WARN] No ground-like detections found. Fallback: keep original detections.")
    class_names = normalized_class_names

class_ids = np.array(list(range(len(class_names))), dtype=np.int32)

labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence in zip(class_names, confidences)
]

"""
Visualize image with supervision useful API
"""
# Get the base filename without extension for output naming
base_filename = Path(img_path).stem

img = cv2.imread(img_path)
detections = sv.Detections(
    xyxy=input_boxes,  # (n, 4)
    mask=masks.astype(bool),  # (n, h, w)
    class_id=class_ids
)

box_annotator = sv.BoxAnnotator()
annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

label_annotator = sv.LabelAnnotator()
annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_filename}_groundingdino_annotated.jpg"), annotated_frame)

mask_annotator = sv.MaskAnnotator()
annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_filename}_grounded_sam2_with_mask.jpg"), annotated_frame)

"""
Dump the results in standard format and save as json files
"""

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

if DUMP_JSON_RESULTS:
    # convert mask into rle format
    mask_rles = [single_mask_to_rle(mask) for mask in masks]

    input_boxes = input_boxes.tolist()
    scores = [float(x) for x in confidences]
    # save the results in standard format
    results = {
        "image_path": img_path,
        "annotations" : [
            {
                "class_name": class_name,
                "bbox": box,
                "segmentation": mask_rle,
                "score": score,
            }
            for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
        ],
        "box_format": "xyxy",
        "img_width": w,
        "img_height": h,
        "has_ground_mask": has_ground_mask,
    }
    
    with open(os.path.join(OUTPUT_DIR, f"{base_filename}_results.json"), "w") as f:
        json.dump(results, f, indent=4)