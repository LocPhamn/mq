import colortrans
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import scipy.sparse
import scipy.sparse.linalg
import scipy.signal
import cv2

# Allow local clone usage without pip install.
_PCT_LOCAL = Path(__file__).resolve().parent / "python-color-transfer"
if _PCT_LOCAL.exists():
    sys.path.insert(0, str(_PCT_LOCAL))
from python_color_transfer.color_transfer import ColorTransfer


def multiplicative_laplacian_guidance(L, eps=1e-6):
    """
    Thay thế ∇²(source_L) bằng Multiplicative Laplacian
    Công thức log-space: 4·ln(Center) - ln(Top) - ln(Bottom) - ln(Left) - ln(Right)
    """
    padded = np.pad(L, 1, mode='edge')

    center = padded[1:-1, 1:-1]
    top    = padded[0:-2, 1:-1]
    bottom = padded[2:,   1:-1]
    left   = padded[1:-1, 0:-2]
    right  = padded[1:-1, 2:  ]

    log_mlap = (4 * np.log(center + eps)
                  - np.log(top    + eps)
                  - np.log(bottom + eps)
                  - np.log(left   + eps)
                  - np.log(right  + eps))
    return log_mlap.astype(np.float64)

def postprocess(img,alpha):
    img = np.clip(img, 0, 255).astype(np.uint8)
    if alpha is not None:
        return np.dstack((img, alpha))
    return img

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

def switch_transfer(src_bgr, ref_bgr):
    """
    chuyển kênh L ref vào src, giữ nguyên A,B của src.
    """
    alpha = src_bgr[..., 3] if src_bgr.shape[2] == 4 else None
    if alpha is not None:
        src_bgr = src_bgr[..., :3]
        
    src_lab = cv2.cvtColor(src_bgr.astype(np.uint8), cv2.COLOR_BGR2Lab)
    ref_lab = cv2.cvtColor(ref_bgr.astype(np.uint8), cv2.COLOR_BGR2Lab)
    
    src_tmp = src_lab.copy()
    src_L = src_lab[:, :, 0]
    ref_L = ref_lab[:, :, 0]
    
    src_lab[:, :, 0] = ref_lab[:, :, 0]
    lab_out = cv2.cvtColor(src_lab, cv2.COLOR_Lab2BGR)
    add_alpha = postprocess(lab_out, alpha)
    return add_alpha
    

def transfer_gauss_region(src_bgr, ref_bgr, mask=None, grid=(2, 2), blur_frac=0.3):
    """
    Apply Gaussian lighting field transfer independently per grid cell.

    Combines the directional awareness of region-based matching with the
    smooth intra-cell correction of the Gaussian field approach:
      1. Divide src and ref into (rows x cols) cells.
      2. Within each cell, apply the additive Gaussian lighting correction:
             new_L = cell_L + (ref_cell_lighting - src_cell_lighting)
         where lighting = GaussianBlur(L, k, sigma).
      3. blur_frac is relative to the cell size, so the kernel scales
         automatically as the grid gets finer.

    Background pixels are filled with the cell's foreground mean before
    blurring to prevent edge bleed. Cells with no foreground pixels are skipped.

    Parameters
    ----------
    src_bgr   : (H, W, 3) uint8      — source image in BGR
    ref_bgr   : (H, W, 3) uint8      — reference region in BGR (may differ in size)
    mask      : (H, W)    uint8|None — foreground mask
    grid      : (int, int)           — (rows, cols) grid divisions
    blur_frac : float                — kernel size as fraction of shorter cell dimension

    Returns
    -------
    (H, W, 3) uint8 BGR — src with per-cell Gaussian lighting matched to ref
    """
    alpha = src_bgr[..., 3] if src_bgr.shape[2] == 4 else None
    if alpha is not None:
        src_bgr = src_bgr[..., :3]

    src_lab = cv2.cvtColor(src_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    ref_lab = cv2.cvtColor(ref_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)

    H, W  = src_bgr.shape[:2]
    ref_L = cv2.resize(ref_lab[:, :, 0], (W, H))
    fg    = (mask > 0) if mask is not None else np.ones((H, W), dtype=bool)

    result_lab = src_lab.copy()
    rows, cols = grid

    for r in range(rows):
        for c in range(cols):
            y1, y2 = r * H // rows, (r + 1) * H // rows
            x1, x2 = c * W // cols, (c + 1) * W // cols

            cell_fg = fg
            if not cell_fg.any():
                continue

            src_cell_L = src_lab[:,:,0]
            ref_cell_L = ref_L 

            cH, cW = src_cell_L.shape
            k      = max(3, int(min(cH, cW) * blur_frac) | 1)
            sigma  = k / 3.0

            # Fill background with fg mean to avoid edge bleed
            src_for_blur = src_cell_L.copy()
            src_for_blur[~cell_fg] = src_cell_L[cell_fg].mean()

            src_lighting = cv2.GaussianBlur(src_for_blur, (k, k), sigma)
            ref_lighting = cv2.GaussianBlur(ref_cell_L,   (k, k), sigma)

            result_lab[:, :, 0] = np.clip(
                src_cell_L + (ref_lighting - src_lighting),
                0.0, 100.0
            )

    result = cv2.cvtColor(result_lab, cv2.COLOR_Lab2BGR)
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    post_processed = postprocess(result, alpha)
    return post_processed


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
    
    img_arr_mlt = PT.transfer_keep_hue(
        src_bgr=rgb_in,
        ref_bgr=img_arr_ref
    )

    img_arr_pdf_reg = postprocess(img_arr_pdf_reg, alpha)
    img_arr_mt = postprocess(img_arr_mt, alpha)
    img_arr_lt = postprocess(img_arr_lt, alpha)
    img_arr_mlt = postprocess(img_arr_mlt, alpha)

    # --- 6. Save ---
    return img_arr_mlt

def transfer_poisson_lum_multiplicative(src_bgr, ref_bgr, mask, strength=0.5, boundary_pad=0):
    H, W = src_bgr.shape[:2]

    # --- Chuẩn bị ảnh ---
    src_bgr_3 = src_bgr[:, :, :3] if src_bgr.ndim == 3 and src_bgr.shape[2] > 3 else src_bgr
    ref_bgr_3 = ref_bgr[:, :, :3] if ref_bgr.ndim == 3 and ref_bgr.shape[2] > 3 else ref_bgr

    src_lab = cv2.cvtColor(src_bgr_3.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    ref_bgr_rs = cv2.resize(ref_bgr_3, (W, H)) if ref_bgr_3.shape[:2] != (H, W) else ref_bgr_3
    ref_lab = cv2.cvtColor(ref_bgr_rs.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)

    src_hsv = cv2.cvtColor(src_bgr_3.astype(np.uint8), cv2.COLOR_BGR2HSV)
    ref_hsv = cv2.cvtColor(ref_bgr_rs.astype(np.uint8), cv2.COLOR_BGR2HSV)

    src_L = src_lab[:, :, 0]                          # [0, 100]
    ref_L = ref_lab[:, :, 0]
    src_S = src_hsv[:, :, 1].astype(np.float64)       # [0, 255]
    ref_S = ref_hsv[:, :, 1].astype(np.float64)

    print(f"src_L range: {src_L.min():.2f} - {src_L.max():.2f}, mean: {src_L.mean():.2f}")
    print(f"ref_L range: {ref_L.min():.2f} - {ref_L.max():.2f}, mean: {ref_L.mean():.2f}")

    # --- Mask ---
    bin_mask = (mask > 0).astype(np.uint8)
    if bin_mask.sum() == 0:
        return postprocess(src_bgr_3.copy(), mask), src_hsv[:, :, 1].copy()

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(bin_mask, kernel, iterations=1)
    boundary_mask = (bin_mask - eroded).clip(0, 1)

    # Optional: thicken boundary band inward by n pixels while keeping it inside mask.
    pad = max(0, int(boundary_pad))
    if pad > 0:
        pad_kernel = np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)
        boundary_mask = cv2.dilate(boundary_mask, pad_kernel, iterations=1)
        boundary_mask = ((boundary_mask > 0) & (bin_mask > 0)).astype(np.uint8)

    inner_mask = ((bin_mask > 0) & (boundary_mask == 0)).astype(np.uint8)
    bin_mask_bool = bin_mask.astype(bool)

    # --- Index sparse pixels ---
    coords_all = np.argwhere(bin_mask > 0)
    n = len(coords_all)
    idx = -np.ones((H, W), dtype=np.int32)
    idx[coords_all[:, 0], coords_all[:, 1]] = np.arange(n)

    ys, xs = coords_all[:, 0], coords_all[:, 1]
    is_inner = inner_mask[ys, xs].astype(bool)
    is_boundary = boundary_mask[ys, xs].astype(bool)
    inn_ids = np.where(is_inner)[0]
    bnd_ids = np.where(is_boundary)[0]

    def solve_multiplicative_channel(src_ch, ref_ch, ch_max):
        """
        Multiplicative Laplacian Poisson:
          w_field  = ref / (src + eps)   ← tỉ lệ mong muốn tại mỗi pixel
          Laplacian Poisson solve: Δw = 0 bên trong, w = w_field tại biên
          → w mượt bên trong, anchored tại viền mask
          Kết quả: result = src * w_solved  (multiplicative apply)
        """
        src_ch = src_ch.astype(np.float64)
        ref_ch = ref_ch.astype(np.float64)

        # Multiplicative guidance field
        w_field = np.clip(ref_ch / (src_ch + 1.0), 1e-6, 1e4)

        # Nếu mask quá mỏng (không có inner), áp w trực tiếp
        if inner_mask.sum() == 0:
            blended = np.clip((1.0 - strength) * src_ch + strength * src_ch * w_field, 0.0, ch_max)
            return np.where(bin_mask_bool, blended, src_ch)

        b = np.zeros(n, dtype=np.float64)
        rows, cols, vals = [], [], []

        # ── Boundary: w_i = w_field_i (Dirichlet BC) ──
        b[bnd_ids] = w_field[ys[bnd_ids], xs[bnd_ids]]
        rows.append(bnd_ids);  cols.append(bnd_ids);  vals.append(np.ones(len(bnd_ids)))

        # ── Inner: Δw = 0  →  4·w_i - Σ w_j = 0 ──
        rows.append(inn_ids);  cols.append(inn_ids);  vals.append(np.full(len(inn_ids), 4.0))

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nys = ys[inn_ids] + dy
            nxs = xs[inn_ids] + dx
            in_bounds = (nys >= 0) & (nys < H) & (nxs >= 0) & (nxs < W)
            nys_c = np.clip(nys, 0, H - 1)
            nxs_c = np.clip(nxs, 0, W - 1)
            j_vals = idx[nys_c, nxs_c]

            in_mask_flag  = in_bounds & (j_vals >= 0)   # neighbor trong mask
            out_mask_flag = in_bounds & (j_vals < 0)    # neighbor ngoài mask → Dirichlet BC

            # off-diagonal -1
            rows.append(inn_ids[in_mask_flag])
            cols.append(j_vals[in_mask_flag])
            vals.append(np.full(in_mask_flag.sum(), -1.0))

            # neighbor ngoài mask: dùng w_field tại đó làm BC
            b[inn_ids[out_mask_flag]] += w_field[nys[out_mask_flag], nxs[out_mask_flag]]

        A = scipy.sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n)
        )

        try:
            x = scipy.sparse.linalg.spsolve(A, b)
            x = np.nan_to_num(x, nan=1.0, posinf=10.0, neginf=0.1)
        except Exception:
            return src_ch.copy()

        # w_solved smooth bên trong mask, = w_field tại biên
        w_solved = np.ones((H, W), dtype=np.float64)
        w_solved[ys, xs] = np.clip(x, 0.01, ch_max)

        # Multiplicative apply + strength blend
        solved = np.clip(src_ch * w_solved, 0.0, ch_max)
        return np.clip((1.0 - strength) * src_ch + strength * solved, 0.0, ch_max)

    # --- Solve L (LAB) ---
    result_L = solve_multiplicative_channel(src_L, ref_L, 100.0)

    # --- Solve S (HSV) ---
    result_S = solve_multiplicative_channel(src_S, ref_S, 255.0)

    # --- Apply L back vào LAB → BGR ---
    result_lab = src_lab.copy()
    result_lab[:, :, 0] = result_L
    out_bgr = cv2.cvtColor(result_lab.astype(np.float32), cv2.COLOR_Lab2BGR)
    out_bgr = np.clip(out_bgr * 255.0, 0, 255).astype(np.uint8)

    # --- Apply S back vào HSV → BGR ---
    out_hsv = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2HSV)
    out_hsv[:, :, 1] = result_S.astype(np.uint8)
    out_bgr = cv2.cvtColor(out_hsv, cv2.COLOR_HSV2BGR)

    return postprocess(out_bgr, mask), result_S.astype(np.uint8)

def transfer_poisson_lum(src_bgr, ref_bgr, mask, strength=0.5, boundary_pad=0):
    H, W = src_bgr.shape[:2]

    src_lab = cv2.cvtColor(src_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    ref_bgr_rs = cv2.resize(ref_bgr, (W, H)) if ref_bgr.shape[:2] != (H, W) else ref_bgr
    ref_lab = cv2.cvtColor(ref_bgr_rs.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)

    src_L = src_lab[:, :, 0]
    ref_L = ref_lab[:, :, 0]

    bin_mask = (mask > 0).astype(np.uint8)
    if bin_mask.sum() == 0:
        return src_bgr.copy()

    kernel        = np.ones((3, 3), np.uint8)
    eroded        = cv2.erode(bin_mask, kernel, iterations=1)
    boundary_mask = (bin_mask - eroded).clip(0, 1)

    # Optional: thicken boundary band inward by n pixels while keeping it inside mask.
    pad = max(0, int(boundary_pad))
    if pad > 0:
        pad_kernel = np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)
        boundary_mask = cv2.dilate(boundary_mask, pad_kernel, iterations=1)
        boundary_mask = ((boundary_mask > 0) & (bin_mask > 0)).astype(np.uint8)

    inner_mask    = ((bin_mask > 0) & (boundary_mask == 0)).astype(np.uint8)

    if inner_mask.sum() == 0:
        result_lab = src_lab.copy()
        result_lab[:, :, 0] = np.where(
            bin_mask.astype(bool),
            (1.0 - strength) * src_L + strength * ref_L,
            src_L
        )
        out = cv2.cvtColor(result_lab, cv2.COLOR_Lab2BGR)
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)

    coords_all = np.argwhere(bin_mask > 0)
    n = len(coords_all)
    idx = -np.ones((H, W), dtype=np.int32)
    idx[coords_all[:, 0], coords_all[:, 1]] = np.arange(n)

    ys, xs      = coords_all[:, 0], coords_all[:, 1]
    is_inner    = inner_mask[ys, xs].astype(bool)
    is_boundary = boundary_mask[ys, xs].astype(bool)
    inn_ids     = np.where(is_inner)[0]
    bnd_ids     = np.where(is_boundary)[0]

    # ── BƯỚC 4 (ĐÃ THAY): Multiplicative Laplacian làm guidance ──────────
    mlap_src_L = multiplicative_laplacian_guidance(src_L)
    # ─────────────────────────────────────────────────────────────────────

    b = np.zeros(n, dtype=np.float64)
    b[bnd_ids] = ref_L[ys[bnd_ids], xs[bnd_ids]].astype(np.float64)
    b[inn_ids] = mlap_src_L[ys[inn_ids], xs[inn_ids]]   # ← dùng mlap thay lap

    all_rows = [bnd_ids, inn_ids]
    all_cols = [bnd_ids, inn_ids]
    all_vals = [np.ones(len(bnd_ids)), np.full(len(inn_ids), 4.0)]

    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nys = ys[inn_ids] + dy
        nxs = xs[inn_ids] + dx
        in_bounds     = (nys >= 0) & (nys < H) & (nxs >= 0) & (nxs < W)
        nys_s         = np.clip(nys, 0, H - 1)
        nxs_s         = np.clip(nxs, 0, W - 1)
        j_vals        = idx[nys_s, nxs_s]
        in_mask_flag  = in_bounds & (j_vals >= 0)
        out_mask_flag = in_bounds & (j_vals < 0)

        all_rows.append(inn_ids[in_mask_flag])
        all_cols.append(j_vals[in_mask_flag])
        all_vals.append(np.full(int(in_mask_flag.sum()), -1.0))

        b[inn_ids[out_mask_flag]] += ref_L[nys[out_mask_flag], nxs[out_mask_flag]].astype(np.float64)

    A = scipy.sparse.csr_matrix(
        (np.concatenate(all_vals),
         (np.concatenate(all_rows), np.concatenate(all_cols))),
        shape=(n, n)
    )

    try:
        x = scipy.sparse.linalg.spsolve(A, b)
        x = np.nan_to_num(x, nan=0.0, posinf=100.0, neginf=0.0)
    except Exception:
        return transfer_keep_hue(src_bgr, ref_bgr, mask=mask)

    result_L         = src_L.copy()
    solved_L         = src_L.copy()
    solved_L[ys, xs] = x
    solved_L         = np.clip(solved_L, 0.0, 100.0)
    result_L         = (1.0 - strength) * src_L + strength * solved_L

    result_lab          = src_lab.copy()
    result_lab[:, :, 0] = result_L
    out = cv2.cvtColor(result_lab, cv2.COLOR_Lab2BGR)
    return postprocess(np.clip(out * 255.0, 0, 255).astype(np.uint8), mask)

def transfer_pyramid_lum(src_bgr, ref_bgr, mask, n_levels=None, cutoff=2):
    """
    Match source luminance to target using Laplacian pyramid decomposition.

    Decomposes both L channels into frequency bands:
      - Fine bands  (high freq): kept from source  → preserves surface texture/detail
      - Coarse bands (low freq): taken from target  → adopts target illumination field

    The number of coarse bands adopted from the target is controlled by `cutoff`.

    Parameters
    ----------
    src_bgr  : (H, W, 3) uint8
    ref_bgr  : (H, W, 3) uint8  — target reference patch, same spatial size as src
    mask     : (H, W)    uint8  — foreground mask (0 or 255)
    n_levels : int | None       — pyramid depth; None = auto from image size
    cutoff   : int              — how many coarse levels to adopt from target (default 2)

    Returns
    -------
    (H, W, 3) uint8 BGR — src with pyramid-harmonised L channel; A/B unchanged
    """
    H, W = src_bgr.shape[:2]

    src_lab = cv2.cvtColor(src_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    ref_bgr_rs = cv2.resize(ref_bgr, (W, H)) if ref_bgr.shape[:2] != (H, W) else ref_bgr
    ref_lab = cv2.cvtColor(ref_bgr_rs.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)

    src_L = src_lab[:, :, 0].astype(np.float64)
    ref_L = ref_lab[:, :, 0].astype(np.float64)

    bin_mask = (mask > 0)
    if not bin_mask.any():
        return src_bgr.copy()

    # Auto pyramid depth: coarsest level ~ 16 px on the short side
    if n_levels is None:
        n_levels = max(2, int(np.floor(np.log2(min(H, W) / 16.0))))
    cutoff = int(np.clip(cutoff, 1, n_levels))

    # Fill background with foreground mean before building pyramid
    # so background values do not bleed into coarser levels near edges
    fg_mean = src_L[bin_mask].mean()
    src_L_pyr = src_L.copy()
    src_L_pyr[~bin_mask] = fg_mean

    ref_fg_mean = ref_L[bin_mask].mean()
    ref_L_pyr = ref_L.copy()
    ref_L_pyr[~bin_mask] = ref_fg_mean

    def gaussian_pyramid(img, levels):
        gp = [img]
        for _ in range(levels):
            gp.append(cv2.pyrDown(gp[-1]))
        return gp

    def laplacian_pyramid(gp):
        lp = []
        for k in range(len(gp) - 1):
            up = cv2.pyrUp(gp[k + 1], dstsize=(gp[k].shape[1], gp[k].shape[0]))
            lp.append(gp[k] - up)
        lp.append(gp[-1])   # coarsest residual
        return lp

    gp_src = gaussian_pyramid(src_L_pyr, n_levels)
    gp_ref = gaussian_pyramid(ref_L_pyr, n_levels)

    lp_src = laplacian_pyramid(gp_src)
    lp_ref = laplacian_pyramid(gp_ref)

    # Blend: fine levels from source, coarse levels from target
    threshold = n_levels - cutoff
    blended_lp = []
    for k in range(n_levels + 1):
        if k < threshold:
            blended_lp.append(lp_src[k])  # fine detail: from source
        else:
            blended_lp.append(lp_ref[k])  # coarse illumination: from target

    # Collapse pyramid back to full resolution
    result_L = blended_lp[-1].copy()
    for k in range(n_levels - 1, -1, -1):
        up = cv2.pyrUp(result_L,
                       dstsize=(blended_lp[k].shape[1], blended_lp[k].shape[0]))
        result_L = up + blended_lp[k]

    result_L = np.clip(result_L, 0.0, 100.0)

    # Soft mask blend: apply result only inside mask with a feathered edge
    k_soft   = max(3, int(min(H, W) * 0.02) | 1)
    soft_mask = cv2.GaussianBlur(
        bin_mask.astype(np.float64), (k_soft, k_soft), k_soft / 3.0
    )
    result_L = src_L * (1.0 - soft_mask) + result_L * soft_mask

    result_lab          = src_lab.copy()
    result_lab[:, :, 0] = result_L.astype(np.float32)
    out = cv2.cvtColor(result_lab, cv2.COLOR_Lab2BGR)
    return  postprocess(np.clip(out * 255.0, 0, 255).astype(np.uint8),mask)

def transfer_bilateral_lum(src_bgr, ref_bgr, mask=None, d=9,
                            sigma_color=75.0, sigma_space=75.0):
    """
    Match luminance via bilateral-filter decomposition (edge-preserving).

    Decomposes the L channel of both source and reference into:
        base   = bilateralFilter(L)          (smooth illumination)
        detail = L - base                    (texture / edges)

    The source base is linearly mapped (mean+std) to match the reference
    base, then recombined with the *original* source detail:

        new_L = transferred_base + source_detail

    Because the bilateral filter is edge-aware, the detail layer cleanly
    captures texture without halo artefacts, so fine surface detail is
    perfectly preserved while overall brightness adapts to the target.

    Parameters
    ----------
    src_bgr     : (H, W, 3) uint8  — source image in BGR
    ref_bgr     : (H, W, 3) uint8  — reference region in BGR (may differ in size)
    mask        : (H, W)    uint8|None — foreground mask for src statistics
    d           : int   — bilateral filter diameter (-1 = auto from sigma_space).
                          Larger values → smoother base, more into 'detail'.
    sigma_color : float — bilateral filter sigma in colour space.
    sigma_space : float — bilateral filter sigma in coordinate space.

    Returns
    -------
    (H, W, 3) uint8 BGR — source with luminance matched, texture preserved
    """
    src_lab = cv2.cvtColor(src_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    ref_lab = cv2.cvtColor(ref_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)

    H, W  = src_bgr.shape[:2]
    ref_L = cv2.resize(ref_lab[:, :, 0], (W, H))
    fg    = (mask > 0) if mask is not None else np.ones((H, W), dtype=bool)

    src_L = src_lab[:, :, 0]

    # Fill background with foreground mean so filter doesn't leak BG values
    src_L_fill = src_L.copy()
    if mask is not None and fg.any():
        src_L_fill[~fg] = src_L[fg].mean()

    # Decompose into base (illumination) + detail (texture)
    src_base = cv2.bilateralFilter(src_L_fill, d, sigma_color, sigma_space)
    ref_base = cv2.bilateralFilter(ref_L,      d, sigma_color, sigma_space)

    src_detail = src_L - src_base          # high-freq texture from source

    # Transfer base: map src_base stats → ref_base stats (foreground only for src)
    src_b_mean = src_base[fg].mean()
    src_b_std  = src_base[fg].std() + 1e-6
    ref_b_mean = ref_base.mean()
    ref_b_std  = ref_base.std() + 1e-6

    transferred_base = (src_base - src_b_mean) * (ref_b_std / src_b_std) + ref_b_mean

    # Recombine: transferred illumination + original texture
    src_lab[:, :, 0] = np.clip(transferred_base + src_detail, 0.0, 100.0)

    result = cv2.cvtColor(src_lab, cv2.COLOR_Lab2BGR)
    return postprocess(np.clip(result * 255.0, 0, 255).astype(np.uint8), mask)

if __name__ == "__main__":
    # content = np.array(Image.open("alpha_clean/earth_mover-152-_png.rf.7b49df990f48fbec091c380813379b6c.png").convert("RGBA"))
    img_arr_in = cv2.imread("alpha_clean/earth_mover-1660-_png.rf.d6e50eb0e44a5508874b8cba74cd7006.png", cv2.IMREAD_UNCHANGED)
    img_arr_ref = cv2.imread("images/bg_images/ch08_20250911121618.png")
    
    if img_arr_in.shape[2] == 4:
        rgb_in = img_arr_in[..., :3]
        alpha = img_arr_in[..., 3]
    else:
        rgb_in = img_arr_in
        alpha = None
    blend_L = transfer_bilateral_lum(rgb_in, img_arr_ref, mask=alpha)
    cv2.imwrite("blend_L.png", blend_L)
    
    