"""Prompts and constants for the fabric-sorting task. Letters A-F are six cut shapes; each has a fixed target basket."""

# Basket names use front-camera image coordinates: near = bottom of the image (robot side), far = top (the pile side).
MAP = dict(A="left_near", B="left_middle", C="left_far", D="right_near", E="right_middle", F="right_far")
FAMILY = ("A", "B", "E")           # shapes the geometry rules can arbitrate between
MAT_BOX = (170, 205, 500, 480)     # x0, y0, x1, y1 of the staging mat on a 640x480 front-camera frame
FLAT_THRESHOLD = 0.80              # P(mat) above which the piece counts as lying flat on the mat

STATE_SUPPORT = """Task context: a dual-arm robot sorts fabric pieces. Pieces come from a pile at the top of
the image, are laid on a black staging mat in the middle of the table, and are then carried into
baskets arranged along the left and right sides of the table.

Image 1 is the FRONT camera at the current frame. near = bottom of the image (robot side),
far = top (the pile). The black staging mat is the dark rectangle in the middle of the table.

Subject: the fabric piece that is on or directly above the black staging mat at this frame.

Image 1: <image>"""

Q_SUPPORT = {"support": {
    "type": "choice",
    "instructions": "What is holding up that fabric piece? Answer only from what you can see.",
    "criteria": {
        "mat": "the mat holds it: it rests on the mat with its whole visible area supported",
        "gripper": "a gripper holds it: it hangs, droops, hangs partly off the mat, or is bunched because the gripper grips it, even if it touches the mat",
        "pile": "it is part of the pile at the top of the image, not on the mat",
        "none": "there is no fabric piece on or directly above the mat",
        "unclear": "the piece cannot be seen well enough to decide"}}}

STATE_SHAPE = """A robot sorts cream-coloured fabric pieces. Image 1 is a crop of the black staging mat; the piece lying
flat on it may be rotated to ANY angle, so judge the outline, not the image direction.
Image 2 is the labelled reference of the six pieces, letters A to F, as seen face up.

Pieces carry small punched holes, and A, B and E have a serrated (saw-tooth) edge with a dark under-layer visible along it.
A: large wedge / blade, one long straight edge, pointed at one end, serrated edge along its upper-left side.
B: wide body with ONE long thin horn rising from the top edge; serrated edge on the lower-left of the body.
C: narrow elongated U-shaped strip, a long oval hole in the middle, two thin tips at the open end.
D: small, compact, irregular rounded patch with no long feature.
E: wide body with a notch at the upper right and one thin tail at the bottom running down-right; serrated edge on the left.
F: large thick C-shaped hook / arc.

Image 1: <image>
Image 2: <image>"""

SHAPES = {
    "A": "wedge / blade with a long straight edge",
    "B": "wide body with one long thin horn rising from its top edge",
    "C": "narrow U-shaped strip with a long oval hole",
    "D": "small compact irregular patch",
    "E": "wide body, notch at upper right, one thin tail at the bottom",
    "F": "large thick C-shaped hook",
    "none": "no fabric piece lying flat on the mat",
    "unknown": "outline not identifiable",
}
Q_SHAPE = {"piece_shape": {"type": "choice", "instructions": "Which of the six shapes is the piece lying on the mat?", "criteria": SHAPES}}

# Pixel rectangles (x0, y0, x1, y1) of the six baskets on the 640x480 front-camera frame. Fixed coordinates: they are
# accurate to roughly 20-40 px and must be re-measured for another camera pose or basket layout.
BASKET_BOXES = {
    "left_far": (0, 155, 195, 245), "left_middle": (0, 232, 195, 335), "left_near": (0, 330, 195, 470),
    "right_far": (438, 115, 600, 235), "right_middle": (450, 190, 612, 320), "right_near": (488, 300, 640, 425),
}
