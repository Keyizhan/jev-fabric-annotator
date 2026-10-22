"""Shape geometry on the staging mat: segment the piece, tell the wedge A from the block-with-appendage pieces B/E,
and tell B from E by chirality.

B and E are mirror images of each other, so their outlines cannot be told apart by overlap. The rule: rotate the piece so
its thin part (horn or tail) points UP, then look at which bottom corner of the body is missing. B lacks the bottom-LEFT
corner, E the bottom-RIGHT one (and the sides swap when the thin part points down, which the rotation takes care of).
"""
import cv2, numpy as np

MIN_MARGIN = 0.12   # smallest corner-fill difference we trust for the B/E decision


def segment(bgr):
    """Cream piece on a dark mat inside a mat crop (BGR). Returns a 0/255 mask of the largest plausible component, or None."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    bright = (gray > 120) & (hsv[:, :, 1] < 110)        # bright and not strongly coloured (excludes pink grippers)
    m = cv2.morphologyEx(bright.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    h, w = m.shape; n, lab, st, _ = cv2.connectedComponentsWithStats(m); best = None
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if area < 1500 or area > 0.25 * h * w: continue
        if x <= 2 or x + bw >= w - 2 or y + bh >= h - 2: continue      # basket / frame strips
        if best is None or area > st[best, 4]: best = i
    return None if best is None else (lab == best).astype(np.uint8) * 255


def clean_segment(bgr):
    """segment() plus a rejection of masks that hug the top edge or span the mat (mat border, not a piece)."""
    m = segment(bgr)
    if m is None: return None
    ys, xs = np.nonzero(m)
    return None if ys.min() <= 3 or (xs.max() - xs.min()) > 0.7 * m.shape[1] else m


def family_features(mask):
    """(aspect ratio of the min-area rectangle, solidity = area / convex-hull area)."""
    cnt = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    (_, _), (w, h), _ = cv2.minAreaRect(cnt); area = cv2.contourArea(cnt)
    return max(w, h) / max(1.0, min(w, h)), area / max(1.0, cv2.contourArea(cv2.convexHull(cnt)))


def _disk(r): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def chirality(mask):
    """Thin-part-up normalisation, then the missing bottom corner. Returns {"side": "L"|"R", "margin", "fill", "angle"} or None."""
    m = (mask > 0).astype(np.uint8); dt = cv2.distanceTransform(m, cv2.DIST_L2, 5); r = float(dt.max())
    core = (dt >= 0.55 * r).astype(np.uint8); body = cv2.dilate(core, _disk(int(0.55 * r) + 2)) & m
    thin = m & (1 - body); n, lab, st, _ = cv2.connectedComponentsWithStats(thin)
    if n < 2: return None
    thin = (lab == 1 + int(np.argmax(st[1:, 4]))).astype(np.uint8)
    by, bx = np.nonzero(body); cb = np.array([bx.mean(), by.mean()])
    ty, tx = np.nonzero(thin); i = int(np.hypot(tx - cb[0], ty - cb[1]).argmax()); v = np.array([tx[i], ty[i]], float) - cb
    ang = float(np.degrees(np.arctan2(v[0], -v[1])))                 # clockwise angle from "up" to the thin part
    size = 3 * int(max(m.shape)); M = cv2.getRotationMatrix2D((float(cb[0]), float(cb[1])), ang, 1.0)   # positive = counter-clockwise
    M[0, 2] += size / 2 - cb[0]; M[1, 2] += size / 2 - cb[1]
    rb = cv2.warpAffine(body, M, (size, size), flags=cv2.INTER_NEAREST); yy, xx = np.nonzero(rb)
    if len(yy) < 50: return None
    x0, x1, y0, y1 = xx.min(), xx.max() + 1, yy.min(), yy.max() + 1; cw, ch = int(0.34 * (x1 - x0)), int(0.34 * (y1 - y0))
    fill = {"BL": float(rb[y1 - ch:y1, x0:x0 + cw].mean()), "BR": float(rb[y1 - ch:y1, x1 - cw:x1].mean())}
    return {"side": "L" if fill["BL"] < fill["BR"] else "R", "margin": abs(fill["BL"] - fill["BR"]), "fill": fill, "angle": ang}


def geometry_vote(bgr_crop):
    """One per-frame vote from a mat crop: "A", "B", "E" or None (no clean piece / not decidable)."""
    m = clean_segment(bgr_crop)
    if m is None: return None
    asp, sol = family_features(m)
    if sol >= 0.82 and asp >= 2.2: return "A"
    if sol <= 0.80 and asp <= 2.1:
        c = chirality(m)
        if c is not None and c["margin"] >= MIN_MARGIN: return "B" if c["side"] == "L" else "E"
    return None
