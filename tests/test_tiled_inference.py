"""Unit tests for tile generation and detection merging."""

import numpy as np
from helmet_yolov10.inference.tiled import class_aware_nms, make_tiles

def test_tiles_cover_bottom_right_corner() -> None:
    assert any(tile.x2 == 1000 and tile.y2 == 1000 for tile in make_tiles(1000, 1000, (640, 640), .25))

def test_nms_keeps_overlapping_different_classes() -> None:
    assert class_aware_nms(np.array([[0,0,10,10],[1,1,11,11]],float), np.array([.9,.8]), np.array([0,1]), .5).tolist() == [0,1]
