API_NAME = "Bag_Detection"
DESCRIPTION = "无人行李"
LABEL = "unattended_luggage"

# COCO 类别（YOLO11n 预训练）
PERSON_CLASS = "person"
LUGGAGE_CLASSES = ("backpack", "suitcase")
DRAW_CLASSES = ("backpack", "suitcase", "person")

DEFAULT_CONF = 0.25
DEFAULT_IOU_THRESHOLD = 0.01
BOX_EXPAND_SCALE = 1.3

# 深度门控（Depth Anything V2 Small）— 默认关闭，init_status 设 use_depth:true 开启
DEFAULT_USE_DEPTH = False
DEFAULT_DEPTH_DIFF_THRESHOLD = 0.25
DEFAULT_DEPTH_MAX_SIDE = 384

# DEFAULT_TIMES = 300
DEFAULT_TIMES = 10
DEFAULT_REQUIRED_FRAMES = 5
DEFAULT_CONTINUE_DELTA = 2.0
