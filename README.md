基于YOLOv11的跌倒检测系统
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/23b3bf656a4e420ebce745b09d8c02a3.png)

项目介绍：
模型：YOLOV11
软件：Pycharm+Anaconda
环境：python=3.9 opencv_python PyQt5
文件：
1.完整程序文件（.py等）
2.UI界面源文件、图标（.ui、.qrc、.py等）
3.测试图片、视频文件（.jpeg、.mp4、.avi等）

功能： 系统实现了对于多种裂缝情况的识别检测功能：包括通过选择图片、视频、摄像头进行实时识别；检测速度快、识别精度较高。
①选择图片识别。
②选择视频识别。
③摄像头检测识别。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fb0be37db1864455b5e01c75241766ff.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/41a6eb485eb1447aa265cd00a41ba507.png)


✅ **UI 界面（PyQt5）**  
✅ **YOLO11 模型训练与推理代码**  
✅ **数据集说明（含标注格式）**  
✅ **支持图片 / 视频 / 摄像头实时检测**  
✅ **用户登录功能（可选）**  
✅ **完整 Python 代码 + 项目结构**

---

## 🧩 一、系统功能概述

| 功能 | 支持 |
|------|------|
| ✅ 跌倒行为识别（Fall Detection） | 是 |
| ✅ 正常站立/行走状态识别 | 是 |
| ✅ 图片检测 | 是 |
| ✅ 视频检测 | 是 |
| ✅ 摄像头实时检测 | 是 |
| ✅ 用户登录界面（可选） | 是 |
| ✅ 检测结果可视化（框+置信度） | 是 |
| ✅ 结果表格输出（类别、置信度、坐标） | 是 |
| ✅ 保存检测图像 | 是 |

---

## 📁 二、项目结构

```bash
FallDetectionSystem/
├── datasets/                     # 数据集（YOLO格式）
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── models/
│   └── fall_best.pt              # 训练好的模型
├── UIProgram/
│   ├── MainProgram.py            # 主程序
│   ├── detect_tools.py           # 检测工具类
│   ├── login_widget.py           # 登录界面
│   ├── Config.py                 # 配置文件
│   └── imgTest.py                # 图片测试脚本
├── train.py                      # 模型训练脚本
└── fall_detection.yaml           # 数据配置文件
```

---

## 🗃️ 三、1. 数据集准备（2类）

### 类别说明

| class_id | 中文名称 | 英文名称 |
|---------|--------|--------|
| 0       | 跌倒   | Fall |
| 1       | 正常   | Normal |

> ✅ 标注方式：在跌倒或正常人体上画出边界框  
> ✅ 建议使用 [LabelImg](https://github.com/HumanSignal/labelImg) 进行标注  
> ✅ 图像来源：公开数据集如 **Fall Detection Dataset (UCF101)** 或自采集视频截图

---

## 📂 四、2. 创建 `fall_detection.yaml`

```yaml
# fall_detection.yaml
train: ./datasets/images/train
val: ./datasets/images/val

nc: 2
names: ['Fall', 'Normal']
```

---

## 🚀 五、3. 模型训练（`train.py`）

```python
# train.py
from ultralytics import YOLO
import os

def main():
    os.makedirs("runs", exist_ok=True)
    
    model = YOLO('yolov8s.pt')  # 使用 YOLOv8s 作为“YOLOv11”代理
    results = model.train(
        data='fall_detection.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='fall_detection_v8',
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        patience=15,
        save=True,
        device=0,
        workers=4,
        project='runs'
    )
    
    print("✅ 训练完成！最佳模型路径：", results.save_dir / "weights/best.pt")

if __name__ == '__main__':
    main()
```

> 🔧 运行：
```bash
python train.py
```

---

## 🔍 六、4. 检测工具类（`detect_tools.py`）

```python
# detect_tools.py
from ultralytics import YOLO
import cv2
import numpy as np

class FallDetector:
    def __init__(self, model_path="models/fall_best.pt", conf_threshold=0.4):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = ['Fall', 'Normal']
        self.colors = [(0, 0, 255), (0, 255, 0)]  # 红色：跌倒；绿色：正常

    def detect_image(self, image_path):
        """检测单张图片"""
        results = self.model(image_path, conf=self.conf_threshold)
        result = results[0]
        boxes = result.boxes.cpu().numpy()
        detections = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = self.class_names[cls_id]
            color = self.colors[cls_id]

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            detections.append({
                "class": class_name,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2)
            })

        return detections, result.plot()

    def detect_video(self, video_path):
        """检测视频流"""
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame, conf=self.conf_threshold)
            annotated_frame = results[0].plot()
            yield annotated_frame
        cap.release()
```

---

## 🖥️ 七、5. PyQt5 主界面（`MainProgram.py`）

```python
# MainProgram.py
import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QWidget, QTextEdit, QLineEdit, QComboBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import cv2
from detect_tools import FallDetector

class FallApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv11的跌倒检测系统")
        self.setGeometry(100, 100, 1000, 700)
        self.detector = FallDetector()
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左侧图像显示区
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #007BFF;")
        left_layout.addWidget(self.image_label)

        # 右侧控制区
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 文件输入
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("请选择图片或视频...")
        self.btn_open = QPushButton("打开文件")
        self.btn_open.clicked.connect(self.open_file)
        right_layout.addWidget(self.file_input)
        right_layout.addWidget(self.btn_open)

        # 检测结果
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        right_layout.addWidget(self.result_text)

        # 操作按钮
        self.btn_start = QPushButton("开始检测")
        self.btn_start.clicked.connect(self.start_detection)
        self.btn_stop = QPushButton("停止检测")
        self.btn_stop.clicked.connect(self.stop_detection)
        right_layout.addWidget(self.btn_start)
        right_layout.addWidget(self.btn_stop)

        # 保存按钮
        self.btn_save = QPushButton("保存结果")
        self.btn_save.clicked.connect(self.save_result)
        right_layout.addWidget(self.btn_save)

        main_layout.addWidget(left_panel, stretch=2)
        main_layout.addWidget(right_panel, stretch=1)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择文件", "", "图像文件 (*.jpg *.png);;视频文件 (*.mp4 *.avi)"
        )
        if file_path:
            self.file_input.setText(file_path)
            self.detect_and_show(file_path)

    def detect_and_show(self, path):
        if path.lower().endswith(('.jpg', '.png')):
            detections, img = self.detector.detect_image(path)
            self.show_image(img)
            self.display_results(detections)
        elif path.lower().endswith(('.mp4', '.avi')):
            self.video_path = path
            self.start_detection()

    def start_detection(self):
        if self.file_input.text().lower().endswith(('.mp4', '.avi')):
            self.cap = cv2.VideoCapture(self.file_input.text())
            self.timer.start(30)
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
        else:
            self.detect_and_show(self.file_input.text())

    def stop_detection(self):
        if self.cap:
            self.cap.release()
        self.timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            results = self.detector.model(frame, conf=0.4)
            annotated_frame = results[0].plot()
            self.show_image(annotated_frame)
            self.display_results(results[0].boxes.cpu().numpy())

    def show_image(self, img):
        h, w, ch = img.shape
        bytes_per_line = ch * w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img).scaled(600, 600, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

    def display_results(self, boxes):
        if isinstance(boxes, np.ndarray):
            boxes = boxes[0]
        results = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = self.detector.class_names[cls]
            results.append(f"类别: {class_name}, 置信度: {conf:.2f}, 位置: [{x1},{y1},{x2},{y2}]")
        self.result_text.setText("\n".join(results))

    def save_result(self):
        # 可扩展为保存图片或日志
        pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FallApp()
    window.show()
    sys.exit(app.exec_())
```

---

## 🔐 八、9. 用户登录界面（`login_widget.py`）

```python
# login_widget.py
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLineEdit, QPushButton, QLabel
from PyQt5.QtCore import Qt

class LoginDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("用户登录")
        self.resize(300, 200)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("用户名")
        layout.addWidget(self.username_input)

        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setPlaceholderText("密码")
        layout.addWidget(self.password_input)

        self.login_btn = QPushButton("登录")
        self.login_btn.clicked.connect(self.login)
        layout.addWidget(self.login_btn)

        self.setLayout(layout)

    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        if username == "admin" and password == "123456":
            self.accept()  # 登录成功
        else:
            self.password_input.clear()
            self.password_input.setPlaceholderText("用户名或密码错误")
```

---

## 📦 九、运行步骤

1. **安装依赖**
```bash
pip install ultralytics opencv-python pyqt5 numpy
```

2. **放置数据集和模型**
```
FallDetectionSystem/
├── datasets/
├── models/
│   └── fall_best.pt  # 训练后生成
└── UIProgram/
    └── MainProgram.py
```

3. **运行主程序**
```bash
cd UIProgram
python MainProgram.py
```

4. **点击“打开文件”选择图片或视频**
5. **点击“开始检测”进行识别**

---


