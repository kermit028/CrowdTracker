"""
初始化追踪点选择对话框
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QImage, QPen, QColor, QFont, QMouseEvent
import cv2
import numpy as np


class InitVideoWidget(QLabel):
    """初始化视频显示组件 - 支持点击选择点"""
    
    point_added = pyqtSignal(float, float)  # x, y
    point_removed = pyqtSignal(int)  # index
    
    def __init__(self, parent=None, max_points=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: #1e1e1e;")
        
        self._frame: np.ndarray = None
        self._qimage: QImage = None
        self._points: list = []  # [(x, y), ...]
        self._max_points = max_points
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._dragging_index = -1
    
    def set_frame(self, frame: np.ndarray):
        """设置帧"""
        self._frame = frame.copy()
        self._update_image()
        self.update()
    
    def get_points(self) -> list:
        """获取选择的点"""
        return self._points.copy()
    
    def clear_points(self):
        """清除所有点"""
        self._points.clear()
        self._dragging_index = -1
        self.update()
    
    def _update_image(self):
        """更新图像"""
        if self._frame is None:
            return
        
        rgb = cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self._qimage = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
    
    def _calculate_transform(self):
        if self._frame is None:
            self._scale = 1.0
            self._offset_x = 0
            self._offset_y = 0
            return
        
        video_h, video_w = self._frame.shape[:2]
        widget_w = self.width()
        widget_h = self.height()
        
        scale_w = widget_w / video_w
        scale_h = widget_h / video_h
        self._scale = min(scale_w, scale_h)
        
        display_w = video_w * self._scale
        display_h = video_h * self._scale
        self._offset_x = (widget_w - display_w) / 2
        self._offset_y = (widget_h - display_h) / 2
    
    def _video_to_widget_coords(self, vx: float, vy: float) -> tuple:
        """视频坐标转控件坐标"""
        if self._frame is None:
            return int(vx), int(vy)
        
        self._calculate_transform()
        
        wx = int(vx * self._scale + self._offset_x)
        wy = int(vy * self._scale + self._offset_y)
        return wx, wy
    
    def _widget_to_video_coords(self, wx: int, wy: int) -> tuple:
        """控件坐标转视频坐标"""
        if self._frame is None:
            return float(wx), float(wy)
        
        self._calculate_transform()
        video_h, video_w = self._frame.shape[:2]
        
        vx = (wx - self._offset_x) / self._scale
        vy = (wy - self._offset_y) / self._scale
        
        vx = max(0, min(video_w - 1, vx))
        vy = max(0, min(video_h - 1, vy))
        
        return vx, vy
    
    def _find_point_at_position(self, wx: int, wy: int, radius: int = 10) -> int:
        """查找指定位置的点索引"""
        for i, (vx, vy) in enumerate(self._points):
            px, py = self._video_to_widget_coords(vx, vy)
            dist = np.sqrt((px - wx)**2 + (py - wy)**2)
            if dist < radius:
                return i
        return -1

    def _interp_point(self, p1, p2, t: float):
        return (
            p1[0] + (p2[0] - p1[0]) * t,
            p1[1] + (p2[1] - p1[1]) * t,
        )
    
    def paintEvent(self, event):
        """绘制"""
        super().paintEvent(event)
        
        if self._qimage is None:
            return
        
        painter = QPainter(self)
        
        # 绘制视频帧
        self._calculate_transform()
        video_h, video_w = self._frame.shape[:2]
        display_w = int(video_w * self._scale)
        display_h = int(video_h * self._scale)
        
        target_rect = QtCore.QRect(
            int(self._offset_x), int(self._offset_y), display_w, display_h
        )
        painter.drawImage(target_rect, self._qimage)
        
        if self._max_points == 4 and len(self._points) >= 2:
            painter.setPen(QPen(QColor(0, 200, 255), 2))
            for i in range(len(self._points) - 1):
                x1, y1 = self._points[i]
                x2, y2 = self._points[i + 1]
                wx1, wy1 = self._video_to_widget_coords(x1, y1)
                wx2, wy2 = self._video_to_widget_coords(x2, y2)
                painter.drawLine(wx1, wy1, wx2, wy2)
            
            if len(self._points) == 4:
                x1, y1 = self._points[-1]
                x2, y2 = self._points[0]
                wx1, wy1 = self._video_to_widget_coords(x1, y1)
                wx2, wy2 = self._video_to_widget_coords(x2, y2)
                painter.drawLine(wx1, wy1, wx2, wy2)
                
                grid_pen = QPen(QColor(0, 200, 255, 128), 1)
                painter.setPen(grid_pen)
                p0, p1, p2, p3 = self._points
                for step in range(1, 10):
                    t = step / 10.0
                    top = self._interp_point(p0, p1, t)
                    bottom = self._interp_point(p3, p2, t)
                    wx1, wy1 = self._video_to_widget_coords(top[0], top[1])
                    wx2, wy2 = self._video_to_widget_coords(bottom[0], bottom[1])
                    painter.drawLine(wx1, wy1, wx2, wy2)
                    
                    left = self._interp_point(p0, p3, t)
                    right = self._interp_point(p1, p2, t)
                    wx1, wy1 = self._video_to_widget_coords(left[0], left[1])
                    wx2, wy2 = self._video_to_widget_coords(right[0], right[1])
                    painter.drawLine(wx1, wy1, wx2, wy2)
                
                painter.setPen(QPen(QColor(0, 200, 255), 2))
        
        # 绘制已选点
        for i, (vx, vy) in enumerate(self._points):
            wx, wy = self._video_to_widget_coords(vx, vy)
            
            # 绘制点
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 255, 0))
            painter.drawEllipse(wx - 6, wy - 6, 12, 12)
            
            if i == self._dragging_index:
                painter.setPen(QPen(QColor(255, 255, 0), 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(wx - 10, wy - 10, 20, 20)
            
            # 绘制编号
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Microsoft YaHei", 10, QFont.Weight.Bold))
            painter.drawText(wx + 10, wy - 10, str(i + 1))
    
    def mousePressEvent(self, event: QMouseEvent):
        """鼠标点击"""
        if self._frame is None:
            return
        
        pos = event.pos()
        wx, wy = pos.x(), pos.y()
        
        # 检查是否点击了已存在的点（右键删除）
        existing_idx = self._find_point_at_position(wx, wy)
        
        if event.button() == Qt.MouseButton.LeftButton:
            if existing_idx >= 0:
                self._dragging_index = existing_idx
                vx, vy = self._widget_to_video_coords(wx, wy)
                self._points[existing_idx] = (vx, vy)
                self.update()
            else:
                if self._max_points is not None and len(self._points) >= self._max_points:
                    return
                # 添加新点
                vx, vy = self._widget_to_video_coords(wx, wy)
                self._points.append((vx, vy))
                self.point_added.emit(vx, vy)
                self.update()
        
        elif event.button() == Qt.MouseButton.RightButton:
            if existing_idx >= 0:
                # 删除点
                del self._points[existing_idx]
                if self._dragging_index == existing_idx:
                    self._dragging_index = -1
                elif self._dragging_index > existing_idx:
                    self._dragging_index -= 1
                self.point_removed.emit(existing_idx)
                self.update()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        if self._frame is None:
            return
        
        if self._dragging_index < 0:
            return
        
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        
        pos = event.pos()
        vx, vy = self._widget_to_video_coords(pos.x(), pos.y())
        self._points[self._dragging_index] = (vx, vy)
        self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging_index = -1
            self.update()


from PyQt6 import QtCore


class InitPointsDialog(QDialog):
    """初始化追踪点选择对话框"""
    
    def __init__(self, frame: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择追踪点 - 请用鼠标左键点击人头位置")
        self.setMinimumSize(800, 600)
        
        self._frame = frame
        self.selected_points: list = []
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # 说明标签
        help_label = QLabel(
            "请用鼠标左键点击要追踪的人头位置，右键点击可删除已选点。\n"
            "选择完成后点击\"开始追踪\"按钮。"
        )
        help_label.setStyleSheet("color: #666; padding: 10px;")
        layout.addWidget(help_label)
        
        # 视频显示
        self.video_widget = InitVideoWidget()
        self.video_widget.set_frame(self._frame)
        layout.addWidget(self.video_widget, stretch=1)
        
        # 信息显示
        self.info_label = QLabel("已选择 0 个追踪点")
        self.info_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        layout.addWidget(self.info_label)
        
        # 按钮
        btn_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton("🗑 清除所有")
        self.clear_btn.clicked.connect(self._on_clear)
        btn_layout.addWidget(self.clear_btn)
        
        btn_layout.addStretch()
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)
        
        self.ok_btn = QPushButton("▶ 开始追踪")
        self.ok_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.ok_btn.clicked.connect(self._on_ok)
        btn_layout.addWidget(self.ok_btn)
        
        layout.addLayout(btn_layout)
        
        # 连接信号
        self.video_widget.point_added.connect(self._on_point_changed)
        self.video_widget.point_removed.connect(self._on_point_changed)
    
    def _on_point_changed(self, *args):
        """点变化"""
        count = len(self.video_widget.get_points())
        self.info_label.setText(f"已选择 {count} 个追踪点")
    
    def _on_clear(self):
        """清除所有点"""
        self.video_widget.clear_points()
        self._on_point_changed()
    
    def _on_ok(self):
        """确认"""
        self.selected_points = self.video_widget.get_points()
        
        if not self.selected_points:
            reply = QMessageBox.question(
                self, "确认",
                "没有选择任何追踪点，将使用默认的示例点。\n是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        
        self.accept()
    
    def get_selected_points(self) -> list:
        """获取选择的点"""
        return self.selected_points


class PerspectiveCorrectionDialog(QDialog):
    def __init__(self, frame: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("画面校正 - 选择四个角点")
        self.setMinimumSize(900, 650)
        
        self._frame = frame
        self.selected_points: list = []
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        help_label = QLabel(
            "请按顺序用鼠标左键点击四个角点：左上 → 右上 → 右下 → 左下。\n"
            "右键点击可删除已选点，选择完成后点击“应用校正”。"
        )
        help_label.setStyleSheet("color: #666; padding: 10px;")
        layout.addWidget(help_label)
        
        self.video_widget = InitVideoWidget(max_points=4)
        self.video_widget.set_frame(self._frame)
        layout.addWidget(self.video_widget, stretch=1)
        
        self.info_label = QLabel("已选择 0 / 4 个角点")
        self.info_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        layout.addWidget(self.info_label)
        
        btn_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton("🗑 清除")
        self.clear_btn.clicked.connect(self._on_clear)
        btn_layout.addWidget(self.clear_btn)
        
        btn_layout.addStretch()
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)
        
        self.ok_btn = QPushButton("✅ 应用校正")
        self.ok_btn.setStyleSheet("""
            QPushButton {
                background-color: #1976D2;
                color: white;
                padding: 8px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
        """)
        self.ok_btn.clicked.connect(self._on_ok)
        btn_layout.addWidget(self.ok_btn)
        
        layout.addLayout(btn_layout)
        
        self.video_widget.point_added.connect(self._on_point_changed)
        self.video_widget.point_removed.connect(self._on_point_changed)
    
    def _on_point_changed(self, *args):
        count = len(self.video_widget.get_points())
        self.info_label.setText(f"已选择 {count} / 4 个角点")
    
    def _on_clear(self):
        self.video_widget.clear_points()
        self._on_point_changed()
    
    def _on_ok(self):
        self.selected_points = self.video_widget.get_points()
        
        if len(self.selected_points) != 4:
            QMessageBox.warning(self, "提示", "请先选择 4 个角点。")
            return
        
        self.accept()
    
    def get_selected_points(self) -> list:
        return self.selected_points
