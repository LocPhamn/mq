import colortrans
import numpy as np
from PIL import Image
from pathlib import Path

import cv2
from python_color_transfer.color_transfer import ColorTransfer

def color_trans(content, reference):
    """
    content: np.array (H, W, 3 or 4)
    reference: np.array (H, W, 3)
    """

    # --- 1. Tách alpha ---
    if content.shape[2] == 4:
        rgb_content = content[..., :3]
        alpha = content[..., 3]
    else:
        rgb_content = content
        alpha = None

    # --- 2. Color transfer (chỉ RGB) ---
    output_rgb = colortrans.transfer_lhm(rgb_content, reference)

    # --- 3. Ghép lại alpha ---
    if alpha is not None:
        # đảm bảo dtype match
        if alpha.dtype != output_rgb.dtype:
            alpha = alpha.astype(output_rgb.dtype)

        output = np.dstack((output_rgb, alpha))
    else:
        output = output_rgb

    return output


def pytorch_color_trans(input_image=None, ref_image=None):

    # --- 1. Load ảnh ---
    img_arr_in = input_image  # giữ alpha
    img_arr_ref = ref_image

    # --- 2. Tách alpha ---
    if img_arr_in.shape[2] == 4:
        rgb_in = img_arr_in[..., :3]
        alpha = img_arr_in[..., 3]
    else:
        rgb_in = img_arr_in
        alpha = None

    # --- 3. Init model ---
    PT = ColorTransfer()

    # --- 4. Transfer ---
    img_arr_pdf_reg = PT.pdf_transfer(
        img_arr_in=rgb_in,
        img_arr_ref=img_arr_ref,
        regrain=True
    )

    img_arr_mt = PT.mean_std_transfer(
        img_arr_in=rgb_in,
        img_arr_ref=img_arr_ref
    )

    img_arr_lt = PT.lab_transfer(
        img_arr_in=rgb_in,
        img_arr_ref=img_arr_ref
    )

    # --- 5. Clip + convert dtype ---
    def postprocess(img):
        img = np.clip(img, 0, 255).astype(np.uint8)
        if alpha is not None:
            return np.dstack((img, alpha))
        return img

    img_arr_pdf_reg = postprocess(img_arr_pdf_reg)
    img_arr_mt = postprocess(img_arr_mt)
    img_arr_lt = postprocess(img_arr_lt)

    # --- 6. Save ---
    return img_arr_pdf_reg

if __name__ == "__main__":
    # content = np.array(Image.open("alpha_clean/earth_mover-152-_png.rf.7b49df990f48fbec091c380813379b6c.png").convert("RGBA"))
    # reference = np.array(Image.open("images/bg_images/ch09_20250911000004.png").convert("RGB"))

    # output = color_trans(content, reference)
    # Image.fromarray(output).save("output.png")
    pass
