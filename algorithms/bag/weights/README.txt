# Bag_Detection 权重

- bag.pt — YOLO11n COCO（person / backpack / suitcase）
- depth/ — Depth Anything V2 Small（见 depth/README.txt）

后处理流程：
  1. YOLO 检测
  2. 无行李 → 结束
  3. use_depth=true 时 → 深度模型 → 同深度的人 → 扩框 IOU
  4. use_depth=false（默认）→ 仅扩框 IOU

init_status 开启深度示例:
  "Bag_Detection": { "conf": 0.4, "use_depth": true }

单图测试:
  cd ~/zq/vision
  python algorithms/bag/test_image.py test.jpg
