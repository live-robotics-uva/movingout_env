import colorsys
import os
from typing import Tuple

RGBTuple = Tuple[float, float, float]

MOVINGOUT_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MAP_PATH = os.path.join(MOVINGOUT_PATH, "maps", "maps_v1")

AVAILABLE_MAPS = {
    "HandOff": "1_HandOff.json",
    "PassOrSplit": "2_PassOrSplit.json",
    "EfficientRoutes": "3_EfficientRoutes.json",
    "PriorityPick": "4_PriorityPick.json",
    "CornerDecision": "5_CornerDecision.json",
    "DistancePriority": "6_DistancePriority.json",
    "TopBottomPriority": "7_TopBottomPriority.json",
    "AdaptiveAssist": "8_AdaptiveAssist.json",
    "LeftRight": "9_LeftRight.json",
    "SingleRotation": "10_SingleRotation.json",
    "FourCorners": "11_FourCorners.json",
    "SequentialRotations": "12_SequentialRotations.json",
    "Simple": "simple.json",
    1000: "1_HandOff.json",
    1001: "2_PassOrSplit.json",
    1002: "3_EfficientRoutes.json",
    1003: "4_PriorityPick.json",
    2000: "5_CornerDecision.json",
    2001: "6_DistancePriority.json",
    2002: "7_TopBottomPriority.json",
    2003: "8_AdaptiveAssist.json",
    2004: "9_LeftRight.json",
    2005: "10_SingleRotation.json",
    2006: "11_FourCorners.json",
    2007: "12_SequentialRotations.json",
}


def get_map_image_path(map_name: str) -> str:
    """Get the full path to the map image."""
    if map_name not in AVAILABLE_MAPS:
        raise ValueError(f"Map '{map_name}' does not exist")
    
    json_filename = AVAILABLE_MAPS[map_name]
    # Replace .json with .png
    png_filename = json_filename.replace('.json', '.png')
    return os.path.join(DEFAULT_MAP_PATH, png_filename)


def rgb(r: float, g: float, b: float) -> RGBTuple:
    return (r / 255.0, g / 255.0, b / 255.0)


def darken_rgb(rgb: RGBTuple) -> RGBTuple:
    """Produce a darker version of a base color."""
    h, l, s = colorsys.rgb_to_hls(*rgb)
    hls_new = (h, max(0, l * 0.9), s)
    return colorsys.hls_to_rgb(*hls_new)


def lighten_rgb(rgb: RGBTuple, times: float = 1.0) -> RGBTuple:
    """Produce a lighter version of a given base color."""
    h, l, s = colorsys.rgb_to_hls(*rgb)
    mult = 1.4**times
    hls_new = (h, 1 - (1 - l) / mult, s)
    return colorsys.hls_to_rgb(*hls_new)


COLORS_RGB = {
    # I'm using Berkeley-branded versions of RGBY from
    # https://brand.berkeley.edu/colors/ (lightened).
    "blue": lighten_rgb(rgb(0x3B, 0x7E, 0xA1), 1.7),  # founder's rock
    "yellow": lighten_rgb(rgb(0xFD, 0xB5, 0x15), 1.7),  # california gold
    "red": lighten_rgb(rgb(0xEE, 0x1F, 0x60), 1.7),  # rose garden
    "green": lighten_rgb(rgb(0x85, 0x94, 0x38), 1.7),  # soybean
    "grey": rgb(162, 163, 175),  # cool grey (not sure which one)
    "brown": rgb(224, 171, 118),  # buff
    "less_light_blue": lighten_rgb(rgb(0x3B, 0x7E, 0xA1), 1.3),  # less light blue
    "less_light_red": lighten_rgb(rgb(255, 102, 102), 1.3),  # less light
    "light_blue": lighten_rgb(rgb(0x7B, 0xC9, 0xC9), 1.7),  # light blue
    "light_red": lighten_rgb(rgb(255, 153, 153), 1.7),  # light red
}

GOAL_LINE_THICKNESS = 0.01
SHAPE_LINE_THICKNESS = 0.015
ROBOT_LINE_THICKNESS = 0.01
# "zoom out" factor when rendering arena; values above 1 will show parts of the
# arena border in allocentric view.
ARENA_ZOOM_OUT = 1.02

ROBOT_RAD = 0.12
ROBOT_MASS = 1.0

PHYS_STEPS = 10
PHYS_ITER = 10

ARENA_SEGMENT_FRICTION = 2.0

CACHE_DISTANCE = False