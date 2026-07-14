# Bag 深度模型（已内置，默认不启用）

路径: `algorithms/bag/weights/depth/Depth-Anything-V2-Small-hf/`

  - config.json
  - preprocessor_config.json
  - model.safetensors

## 如何开启深度后处理

在 init_status.json 里对 Bag_Detection 显式设置:

  "Bag_Detection": {
    "conf": 0.4,
    "use_depth": true,
    "depth_diff_thresh": 0.25,
    "depth_max_side": 384,
    "depth_half": false
  }

## 如何关闭（退回纯 2D 扩框+IOU）

  "Bag_Detection": {
    "conf": 0.4,
    "use_depth": false
  }

或不写 use_depth（默认就是关闭）。

也可全局环境变量（所有摄像头）:

  export EDGE_BAG_USE_DEPTH=1   # 开启
  export EDGE_BAG_USE_DEPTH=0   # 关闭

单图测试:

  python algorithms/bag/test_image.py test.jpg           # 无深度
  python algorithms/bag/test_image.py test.jpg --depth     # 有深度
