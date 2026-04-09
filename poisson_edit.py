import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import cv2


def poisson_image_editing(source, target, offset, alpha = 0):

    source_mask = source[:, :, 3]
    source_mask = (source_mask > 0).astype(np.uint8)

    source = source[:, :, :3]

    target = target.astype(np.float64)
    source = source.astype(np.float64)

    blended_image = target.copy()

    dx, dy = offset[0], offset[1]

    h, w = source_mask.shape
    target_mask = np.zeros(target.shape[:2], dtype=np.uint8)
    target_mask[dx:dx+h, dy:dy+w] = source_mask

    channels = source.shape[2]
    n = np.sum(target_mask > 0)

    grid = np.zeros_like(target_mask, dtype=np.int32)
    ys, xs = np.where(target_mask > 0)
    grid[ys, xs] = np.arange(1, len(ys) + 1)

    for channel in range(channels):
        ys, xs = np.where(grid > 0)

        # ✔️ dùng sparse
        A = lil_matrix((n, n))
        B = np.zeros(n)

        s = source[:, :, channel]
        t = target[:, :, channel]

        for i in range(n):
            y, x = ys[i], xs[i]
            A[i, i] = 4

            # NORTH
            if target_mask[y-1, x] > 0:
                A[i, grid[y-1, x]-1] = -1
            else:
                B[i] += t[y-1, x]

            # SOUTH
            if target_mask[y+1, x] > 0:
                A[i, grid[y+1, x]-1] = -1
            else:
                B[i] += t[y+1, x]

            # WEST
            if target_mask[y, x-1] > 0:
                A[i, grid[y, x-1]-1] = -1
            else:
                B[i] += t[y, x-1]

            # EAST
            if target_mask[y, x+1] > 0:
                A[i, grid[y, x+1]-1] = -1
            else:
                B[i] += t[y, x+1]

            sy, sx = y - dx, x - dy

            # ✔️ tránh crash index
            if not (0 <= sy < h and 0 <= sx < w):
                continue

            v = 0
            center_t = t[y, x]
            center_s = s[sy, sx]

            # NORTH
            if (0 <= sy-1 < h) and source_mask[sy-1, sx] > 0:
                v += center_s - s[sy-1, sx]
            else:
                v += center_t - t[y-1, x]

            # SOUTH
            if (0 <= sy+1 < h) and source_mask[sy+1, sx] > 0:
                v += center_s - s[sy+1, sx]
            else:
                v += center_t - t[y+1, x]

            # WEST
            if (0 <= sx-1 < w) and source_mask[sy, sx-1] > 0:
                v += center_s - s[sy, sx-1]
            else:
                v += center_t - t[y, x-1]

            # EAST
            if (0 <= sx+1 < w) and source_mask[sy, sx+1] > 0:
                v += center_s - s[sy, sx+1]
            else:
                v += center_t - t[y, x+1]

            B[i] += v

        # ✔️ convert đúng cách
        A_sparse = A.tocsr()
        X = spsolve(A_sparse, B)

        for i in range(n):
            y, x = ys[i], xs[i]
            blended_image[y, x, channel] = X[i]

    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)

    return blended_image


def poisson_image_editing_mixing_gradients(
	source,
	target,
	target_mask,
	offset=(0, 0),
	source_mask=None,
	use_source_alpha=True,
	source_opacity=1.0,
	gradient_mode="source",
	mask_threshold=16,
):
	"""
	Poisson image editing with mixed gradients.

	Parameters
	----------
	source : np.ndarray
		Source image, expected shape (H, W, 4) for RGBA. RGB is also accepted.
	target : np.ndarray
		Target image (canvas), shape (Ht, Wt, 3) or (Ht, Wt, 4).
	target_mask : np.ndarray
		Binary mask on the target domain (Omega), shape (H, W).
	offset : tuple[int, int]
		(dy, dx) position offset from source coordinates to target coordinates.
		A source pixel (ys, xs) maps to target pixel (ys + dy, xs + dx).
	source_mask : np.ndarray | None
		Optional binary source mask in source coordinates, shape (Hs, Ws).
		If None and source is RGBA, alpha > 0 is used.
	use_source_alpha : bool
		If True and source has alpha channel, alpha contributes to effective mask.
	source_opacity : float
		Opacity of pasted source effect in [0, 1].
		0 means keep target unchanged, 1 means full Poisson result.
	gradient_mode : str
		"source" keeps source gradients only (sharper, less washed-out).
		"mixed" keeps stronger gradient between source and target.
	mask_threshold : int
		Threshold applied when source_mask is not bool or inferred from alpha.

	Returns
	-------
	np.ndarray
		Blended image in uint8 with same shape as target.
	"""
	source = np.array(source)
	target = np.array(target)
	target_mask = np.array(target_mask).astype(bool)

	if source.ndim != 3 or target.ndim != 3:
		raise ValueError("source and target must be HxWxC arrays")
	ht, wt = target.shape[:2]
	hs, ws = source.shape[:2]

	if target_mask.shape != (ht, wt):
		raise ValueError("target_mask must have shape (H, W)")
	if not (0.0 <= float(source_opacity) <= 1.0):
		raise ValueError("source_opacity must be in [0, 1]")
	if gradient_mode not in ("source", "mixed"):
		raise ValueError("gradient_mode must be 'source' or 'mixed'")
	source_opacity = float(source_opacity)
	mask_threshold = int(mask_threshold)

	dy, dx = int(offset[0]), int(offset[1])

	source_rgb = source[..., :3].astype(np.float64)
	target_base = target.astype(np.float64)
	target_work = target.astype(np.float64).copy()

	if source_mask is None:
		if source.shape[2] == 4 and use_source_alpha:
			source_mask = source[..., 3] > mask_threshold
		else:
			source_mask = np.ones((hs, ws), dtype=bool)
	else:
		source_mask = np.asarray(source_mask)
		if source_mask.shape != (hs, ws):
			raise ValueError("source_mask must have shape (Hs, Ws)")
		if source_mask.dtype != np.bool_:
			source_mask = source_mask > mask_threshold

	# Map each target pixel to source coordinates (without cyclic wrap-around).
	yy, xx = np.indices((ht, wt))
	sy = yy - dy
	sx = xx - dx
	inside_source = (sy >= 0) & (sy < hs) & (sx >= 0) & (sx < ws)

	valid_source_on_target = np.zeros((ht, wt), dtype=bool)
	if np.any(inside_source):
		valid_source_on_target[inside_source] = source_mask[sy[inside_source], sx[inside_source]]

	# Effective Omega keeps only pixels requested in target mask and available in source.
	mask = target_mask & valid_source_on_target

	# Remove borders to avoid out-of-bounds neighbors in 4-neighborhood operations.
	mask[0, :] = False
	mask[-1, :] = False
	mask[:, 0] = False
	mask[:, -1] = False

	ys, xs = np.nonzero(mask)
	n = len(xs)
	if n == 0:
		return target.astype(np.uint8)

	grid = -np.ones((ht, wt), dtype=np.int32)
	grid[ys, xs] = np.arange(n, dtype=np.int32)

	neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
	A = lil_matrix((n, n), dtype=np.float64)

	for i, (y, x) in enumerate(zip(ys, xs)):
		A[i, i] = 4.0
		for ddy, ddx in neighbors:
			ny, nx = y + ddy, x + ddx
			j = grid[ny, nx]
			if j >= 0:
				A[i, j] = -1.0

	A = A.tocsr()

	channels_to_solve = min(3, target_work.shape[2], source_rgb.shape[2])
	for c in range(channels_to_solve):
		s = source_rgb[..., c]
		t = target_work[..., c]
		B = np.zeros(n, dtype=np.float64)

		for i, (y, x) in enumerate(zip(ys, xs)):
			b = 0.0
			v = 0.0
			sy0 = y - dy
			sx0 = x - dx
			tp = t[y, x]
			sp = s[sy0, sx0]

			for ddy, ddx in neighbors:
				ny, nx = y + ddy, x + ddx

				if not mask[ny, nx]:
					b += t[ny, nx]

				grad_t = tp - t[ny, nx]
				nsy, nsx = ny - dy, nx - dx
				if 0 <= nsy < hs and 0 <= nsx < ws:
					grad_s = sp - s[nsy, nsx]
					if gradient_mode == "mixed":
						v += grad_t if abs(grad_t) > abs(grad_s) else grad_s
					else:
						v += grad_s
				else:
					v += grad_t

			B[i] = b + v

		X = spsolve(A, B)
		if source_opacity == 1.0:
			target_work[ys, xs, c] = X
		else:
			target_work[ys, xs, c] = (
				(1.0 - source_opacity) * target_base[ys, xs, c]
				+ source_opacity * X
			)

	blended = np.clip(target_work, 0, 255).astype(np.uint8)
	return blended

if __name__ == "__main__":

    blended_image = poisson_image_editing(
        source=None,
        target=None,
        source_mask=None,
        target_mask=None,
        offset=(200, 500),
    )
    cv2.imwrite("blended.png", blended_image)