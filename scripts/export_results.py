from openpyxl import Workbook

wb = Workbook()
ws = wb.active

ws.title = "Model Comparison"

ws.append([
    "Model",
    "mAP50",
    "mAP50-95",
    "Precision",
    "Recall",
    "FPS",
    "Model Size (MB)"
])

ws.append(["YOLO11", 0.2192, 0.1361, 0.5321, 0.2691, 68.73, 5.3])
ws.append(["Faster R-CNN", 0.3610, 0.2098, 0.3610, 0.3023, 12.70, 159.0])
ws.append(["RetinaNet", 0.2279, 0.1317, 0.2279, 0.2465, 12.90, 124.0])
ws.append(["SSD300", 0.0963, 0.0468, 0.0963, 0.0799, 47.55, 96.0])
ws.append(["FCOS", 0.1974, 0.1146, 0.1974, 0.2123, 14.04, 123.0])

wb.save("results/model_comparison.xlsx")

print("Excel сохранён.")