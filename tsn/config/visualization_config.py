# -*- coding: utf-8 -*-

"""
@date: 2020/10/19 上午9:33
@file: visualization_config.py
@author: zj
@description: 
"""

from yacs.config import CfgNode as CN


def add_visualization_config(_C):
    # visualization configs.
    # ---------------------------------------------------------------------------- #
    # Visualization options
    # ---------------------------------------------------------------------------- #
    _C.VISUALIZATION = CN()
    # Run model in DEMO mode.
    _C.VISUALIZATION.ENABLE = False

    # ---------------------------------------------------------------------------- #
    # Manager options
    # ---------------------------------------------------------------------------- #
    # Specify a camera device as input. This will be prioritized
    # over input manager if set.
    # If -1, use input manager instead.
    _C.VISUALIZATION.WEBCAM = -1
    # Path to input manager for demo.
    _C.VISUALIZATION.INPUT_VIDEO = ""
    # Custom width for reading input manager data.
    _C.VISUALIZATION.DISPLAY_WIDTH = 0
    # Custom height for reading input manager data.
    _C.VISUALIZATION.DISPLAY_HEIGHT = 0
    # Frames per second rate for writing to output manager file.
    # If not set (-1), use fps rate from input file.
    _C.VISUALIZATION.OUTPUT_FPS = -1
    # If specified, the visualized outputs will be written this a manager file of
    # this path. Otherwise, the visualized outputs will be displayed in a window.
    _C.VISUALIZATION.OUTPUT_FILE = ""
    # Number of overlapping frames between 2 consecutive clips.
    # Increase this number for more frequent action predictions.
    # The number of overlapping frames cannot be larger than
    # half of the sequence length `cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE`
    _C.VISUALIZATION.BUFFER_SIZE = 0
    # Draw visualization frames in [keyframe_idx - CLIP_VIS_SIZE, keyframe_idx + CLIP_VIS_SIZE] inclusively.
    _C.VISUALIZATION.CLIP_VIS_SIZE = 10

    # Whether to run in with multi-threaded manager reader.
    _C.VISUALIZATION.THREAD_ENABLE = False
    # Take one clip for every `DEMO.NUM_CLIPS_SKIP` + 1 for prediction and visualization.
    # This is used for fast demo speed by reducing the prediction/visualiztion frequency.
    # If -1, take the most recent read clip for visualization. This mode is only supported
    # if `DEMO.THREAD_ENABLE` is set to True.
    _C.VISUALIZATION.NUM_CLIPS_SKIP = 0

    # ---------------------------------------------------------------------------- #
    # Visualizer options
    # ---------------------------------------------------------------------------- #
    # Number of processes to run manager visualizer.
    _C.VISUALIZATION.NUM_VIS_INSTANCES = 2
    # This is chosen based on distribution of examples in
    # each classes in AVA dataset.
    _C.VISUALIZATION.COMMON_CLASS_NAMES = [
        "watch (a person)",
        "talk to (e.g., self, a person, a group)",
        "listen to (a person)",
        "touch (an object)",
        "carry/hold (an object)",
        "walk",
        "sit",
        "lie/sleep",
        "bend/bow (at the waist)",
    ]
    # Path to a json file providing class_name - id mapping
    # in the format {"class_name1": id1, "class_name2": id2, ...}.
    _C.VISUALIZATION.LABEL_FILE_PATH = ""
    # Colormap to for text boxes and bounding boxes colors
    _C.VISUALIZATION.COLORMAP = "Pastel2"
    # Threshold for common class names.
    _C.VISUALIZATION.COMMON_CLASS_THRES = 0.7
    # Theshold for uncommon class names. This will not be
    # used if `_C.VISUALIZATION.COMMON_CLASS_NAMES` is empty.
    _C.VISUALIZATION.UNCOMMON_CLASS_THRES = 0.3

    # ---------------------------------------------------------------------------- #
    # Predictor options
    # ---------------------------------------------------------------------------- #
    # Input format from demo manager reader ("RGB" or "BGR").
    _C.VISUALIZATION.INPUT_FORMAT = "BGR"
    # Visualize with top-k predictions or predictions above certain threshold(s).
    # Option: {"thres", "top-k"}
    _C.VISUALIZATION.VIS_MODE = "thres"
    # Slow-motion rate for the visualization. The visualized portions of the
    # manager will be played `_C.VISUALIZATION.SLOWMO` times slower than usual speed.
    _C.VISUALIZATION.SLOWMO = 1
