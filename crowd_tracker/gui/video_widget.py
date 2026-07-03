"""
视频显示组件 - 支持追踪点显示和交互
"""
from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QRect
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QMouseEvent, QKeyEvent
import cv2
import numpy as np
from typing import Optional, Tuple, Dict
from core.tracker import Track


class VideoWidget(QWidget):
    """视频显示组件"""
    
    # 信号定义
    track_corrected = pyqtSignal(int, int, float, float)  # track_id, frame_idx, x, y
    frame_clicked = pyqtSignal(float, float)  # x, y (视频坐标)
    add_track_requested = pyqtSignal(float, float, int)  # x, y, frame_idx
    remove_track_requested = pyqtSignal(int)  # track_id
    add_footprint_requested = pyqtSignal(float, float, int)  # x, y, frame_idx
    remove_footprint_requested = pyqtSignal(int)  # footprint_id
    end_track_requested = pyqtSignal(int, int, float, float)  # track_id, frame_idx, x, y
    restore_track_requested = pyqtSignal(int, int, float, float)  # track_id, frame_idx, x, y
    color_picked = pyqtSignal(int, int, tuple)  # x, y, (r, g, b) - 取色信号
    vertical_lines_changed = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(640, 480)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)  # 启用鼠标追踪
        
        # 当前帧图像
        self._current_frame: Optional[np.ndarray] = None
        
        # 显示设置
        self._show_track_points = True
        self._show_trajectory = True
        self._show_track_ids = True
        self._show_footprints = True
        self._trajectory_frame_count = 10
        self._fps = 30
        
        # 追踪数据
        self._current_frame_idx = 0
        self._tracks: Dict[int, Track] = {}
        self._current_positions: Dict[int, Tuple[float, float]] = {}
        self._disappearing_tracks: Dict[int, Track] = {}
        self._disappearing_positions: Dict[int, Tuple[float, float]] = {}
        self._footprint_points = []
        self._marker_color = QColor(255, 140, 0)
        
        # 交互状态
        self._is_dragging = False
        self._dragged_track_id: Optional[int] = None
        self._ctrl_pressed = False
        
        # 编辑模式
        self._add_mode_enabled = True
        self._del_mode_enabled = True
        self._add_footprint_mode_enabled = True
        self._del_footprint_mode_enabled = True
        self._end_mode_enabled = True
        
        # 消失点显示颜色
        self._end_point_color = (0, 255, 0)
        
        # 缩放和平移
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0
        
        # 选中高亮
        self._selected_track_id: Optional[int] = None
        self._selected_footprint_id: Optional[int] = None
        
        # 取色器模式
        self._pick_color_mode = False
        self._pick_color_circle_pos: Optional[Tuple[int, int]] = None  # 视频坐标
        self._pick_color_circle_radius: int = 20
        
        self._show_vertical_lines = True
        self._vertical_line_xs = []
        self._vertical_line_color = (255, 0, 0, 127)
        self._analysis_trajectory_positions: Dict[int, list] = {}
    
    def set_frame(self, frame: np.ndarray, frame_idx: int):
        """设置当前显示的帧"""
        self._current_frame = frame.copy()
        self._current_frame_idx = frame_idx
        self.update()
    
    def set_tracks(self, tracks: Dict[int, Track], current_positions: Dict[int, Tuple[float, float]],
                   disappearing_tracks: Dict[int, Track] = None, disappearing_positions: Dict[int, Tuple[float, float]] = None):
        """设置当前追踪数据"""
        self._tracks = tracks
        self._current_positions = current_positions
        self._disappearing_tracks = disappearing_tracks or {}
        self._disappearing_positions = disappearing_positions or {}
        self.update()
    
    def set_footprints(self, footprint_points: list):
        self._footprint_points = list(footprint_points)
        current_ids = {item["id"] for item in self._footprint_points}
        if self._selected_footprint_id is not None and self._selected_footprint_id not in current_ids:
            self._selected_footprint_id = None
        self.update()

    def set_analysis_trajectory_positions(self, trajectory_positions: Dict[int, list]):
        self._analysis_trajectory_positions = {
            int(track_id): list(points)
            for track_id, points in (trajectory_positions or {}).items()
        }
        self.update()
    
    def set_fps(self, fps: int):
        """设置帧率"""
        self._fps = fps
    
    def set_trajectory_frame_count(self, frame_count: int):
        """设置轨迹显示帧数"""
        self._trajectory_frame_count = max(1, int(frame_count))
        self.update()
    
    def set_show_track_points(self, show: bool):
        """设置是否显示追踪点"""
        self._show_track_points = show
        self.update()
    
    def set_show_trajectory(self, show: bool):
        """设置是否显示轨迹"""
        self._show_trajectory = show
        self.update()
    
    def set_show_track_ids(self, show: bool):
        """设置是否显示编号"""
        self._show_track_ids = show
        self.update()

    def set_show_footprints(self, show: bool):
        """设置是否显示脚印点"""
        self._show_footprints = show
        if not show:
            self._selected_footprint_id = None
        self.update()
    
    def set_selected_track(self, track_id: Optional[int]):
        """设置选中的追踪点"""
        self._selected_track_id = track_id
        self.update()
    
    def set_add_mode_enabled(self, enabled: bool):
        """设置是否启用增加点模式"""
        self._add_mode_enabled = enabled
    
    def set_del_mode_enabled(self, enabled: bool):
        """设置是否启用删除点模式"""
        self._del_mode_enabled = enabled
    
    def set_add_footprint_mode_enabled(self, enabled: bool):
        """设置是否启用增加脚印点模式"""
        self._add_footprint_mode_enabled = enabled
    
    def set_del_footprint_mode_enabled(self, enabled: bool):
        """设置是否启用删除脚印点模式"""
        self._del_footprint_mode_enabled = enabled
    
    def set_end_mode_enabled(self, enabled: bool):
        """设置是否启用结束点模式"""
        self._end_mode_enabled = enabled
    
    def set_end_point_color(self, color: Tuple[int, int, int]):
        """设置消失点显示颜色"""
        self._end_point_color = color
        self.update()
    
    def set_pick_color_mode(self, enabled: bool):
        """设置取色器模式"""
        self._pick_color_mode = enabled
        if not enabled:
            self._pick_color_circle_pos = None
        self.update()
    
    def set_pick_color_circle(self, pos: Optional[Tuple[int, int]], radius: int):
        """设置取色参考圆位置和半径"""
        self._pick_color_circle_pos = pos
        self._pick_color_circle_radius = radius
        self.update()
    
    def set_show_vertical_lines(self, show: bool):
        self._show_vertical_lines = show
        self.update()
    
    def set_vertical_line_positions(self, xs: list):
        self._vertical_line_xs = sorted([float(x) for x in xs])
        self.update()
    
    def get_vertical_line_positions(self) -> list:
        return self._vertical_line_xs.copy()
    
    def clear_vertical_lines(self):
        self._vertical_line_xs = []
        self.vertical_lines_changed.emit(self._vertical_line_xs.copy())
        self.update()
    
    def _calculate_transform(self):
        """计算视频到控件的变换参数"""
        if self._current_frame is None:
            self._scale = 1.0
            self._offset_x = 0
            self._offset_y = 0
            return
        
        video_h, video_w = self._current_frame.shape[:2]
        widget_w = self.width()
        widget_h = self.height()
        
        scale_w = widget_w / video_w
        scale_h = widget_h / video_h
        self._scale = min(scale_w, scale_h)
        
        display_w = video_w * self._scale
        display_h = video_h * self._scale
        self._offset_x = (widget_w - display_w) / 2
        self._offset_y = (widget_h - display_h) / 2
    
    def _video_to_widget_coords(self, vx: float, vy: float) -> Tuple[int, int]:
        """将视频坐标转换为控件坐标"""
        self._calculate_transform()
        wx = int(vx * self._scale + self._offset_x)
        wy = int(vy * self._scale + self._offset_y)
        return wx, wy
    
    def _widget_to_video_coords(self, wx: int, wy: int) -> Optional[Tuple[float, float]]:
        """将控件坐标转换为视频坐标"""
        self._calculate_transform()
        
        if self._current_frame is None:
            return None
        
        video_h, video_w = self._current_frame.shape[:2]
        vx = (wx - self._offset_x) / self._scale
        vy = (wy - self._offset_y) / self._scale
        
        if vx < 0 or vx >= video_w or vy < 0 or vy >= video_h:
            return None
        
        return vx, vy
    
    def _is_inside_video_area(self, wx: int, wy: int) -> bool:
        """检查控件坐标是否在视频显示区域内"""
        self._calculate_transform()
        
        if self._current_frame is None:
            return False
        
        video_h, video_w = self._current_frame.shape[:2]
        vx = (wx - self._offset_x) / self._scale
        vy = (wy - self._offset_y) / self._scale
        
        return 0 <= vx < video_w and 0 <= vy < video_h
    
    def _get_pixel_color(self, vx: int, vy: int) -> Optional[Tuple[int, int, int]]:
        """获取指定视频坐标的像素颜色 (BGR转RGB)"""
        if self._current_frame is None:
            return None
        
        h, w = self._current_frame.shape[:2]
        if 0 <= vx < w and 0 <= vy < h:
            bgr = self._current_frame[int(vy), int(vx)]
            return (int(bgr[2]), int(bgr[1]), int(bgr[0]))  # BGR to RGB
        return None
    
    def _find_track_at_position(self, wx: int, wy: int, radius: int = 10) -> Optional[int]:
        """查找指定控件位置附近的追踪点"""
        coords = self._widget_to_video_coords(wx, wy)
        if coords is None:
            return None
        vx, vy = coords
        
        min_dist = float('inf')
        closest_track = None
        
        for track_id, pos in self._current_positions.items():
            dist = np.sqrt((pos[0] - vx)**2 + (pos[1] - vy)**2)
            if dist < radius / self._scale and dist < min_dist:
                min_dist = dist
                closest_track = track_id
        
        for track_id, pos in self._disappearing_positions.items():
            dist = np.sqrt((pos[0] - vx)**2 + (pos[1] - vy)**2)
            if dist < radius / self._scale and dist < min_dist:
                min_dist = dist
                closest_track = track_id
        
        return closest_track
    
    def _find_footprint_at_position(self, wx: int, wy: int, radius: int = 10) -> Optional[int]:
        if not self._show_footprints:
            return None
        coords = self._widget_to_video_coords(wx, wy)
        if coords is None:
            return None
        vx, vy = coords
        
        min_dist = float('inf')
        closest_id = None
        for footprint in self._footprint_points:
            dist = np.sqrt((footprint["x"] - vx) ** 2 + (footprint["y"] - vy) ** 2)
            if dist < radius / self._scale and dist < min_dist:
                min_dist = dist
                closest_id = footprint["id"]
        return closest_id
    
    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        if self._current_frame is None:
            return
        
        self._calculate_transform()
        
        video_h, video_w = self._current_frame.shape[:2]
        display_w = int(video_w * self._scale)
        display_h = int(video_h * self._scale)
        
        # 绘制视频帧
        target_rect = QRect(
            int(self._offset_x),
            int(self._offset_y),
            display_w,
            display_h
        )
        
        # 将 OpenCV BGR 转为 RGB 显示
        rgb_image = cv2.cvtColor(self._current_frame, cv2.COLOR_BGR2RGB)
        from PyQt6.QtGui import QImage
        qimage = QImage(rgb_image.data, video_w, video_h, rgb_image.strides[0], QImage.Format.Format_RGB888)
        painter.drawImage(target_rect, qimage)
        
        if self._show_vertical_lines and self._vertical_line_xs:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            pen = QPen(QColor(*self._vertical_line_color))
            pen.setWidth(3)
            painter.setPen(pen)
            
            top_y = 0
            bottom_y = video_h - 1
            for vx in self._vertical_line_xs:
                x1, y1 = self._video_to_widget_coords(vx, top_y)
                x2, y2 = self._video_to_widget_coords(vx, bottom_y)
                painter.drawLine(x1, y1, x2, y2)
                
                base = int(12 * self._scale)
                painter.drawLine(x2 - base, y2, x2, y2 - base)
                painter.drawLine(x2, y2 - base, x2 + base, y2)
        
        if self._show_footprints:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            for footprint in self._footprint_points:
                wx, wy = self._video_to_widget_coords(footprint["x"], footprint["y"])
                size = 10
                if footprint["id"] == self._selected_footprint_id:
                    painter.setPen(QPen(QColor(255, 255, 0), 2))
                    painter.setBrush(QColor(255, 255, 0, 80))
                    painter.drawEllipse(wx - 9, wy - 9, 18, 18)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(*footprint["color"]))
                painter.drawEllipse(wx - size // 2, wy - size // 2, size, size)
                
                footprint_track_id = footprint.get("track_id")
                if footprint_track_id is not None:
                    painter.setPen(QColor(255, 255, 255))
                    painter.setFont(QFont("Microsoft YaHei", 9))
                    painter.drawText(wx + 8, wy - 8, str(footprint_track_id))
        
        # 绘制轨迹线
        if self._show_trajectory:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            for track_id, track in self._tracks.items():
                if self._analysis_trajectory_positions.get(track_id):
                    recent = self._analysis_trajectory_positions[track_id]
                else:
                    recent = track.get_recent_positions(
                        self._current_frame_idx, 
                        self._trajectory_frame_count
                    )
                
                if len(recent) > 1:
                    color = QColor(*track.color)
                    pen = QPen(color)
                    pen.setWidth(2)
                    painter.setPen(pen)
                    
                    for i in range(len(recent) - 1):
                        _, pos1 = recent[i]
                        _, pos2 = recent[i + 1]
                        x1, y1 = self._video_to_widget_coords(pos1[0], pos1[1])
                        x2, y2 = self._video_to_widget_coords(pos2[0], pos2[1])
                        painter.drawLine(x1, y1, x2, y2)
        
        # 绘制追踪点
        if self._show_track_points:
            for track_id, pos in self._current_positions.items():
                wx, wy = self._video_to_widget_coords(pos[0], pos[1])
                track = self._tracks.get(track_id)
                if track is None:
                    continue
                
                is_corrected = track.is_manual_corrected.get(self._current_frame_idx, False)
                
                if track_id == self._selected_track_id:
                    painter.setPen(QPen(QColor(255, 255, 0), 3))
                    painter.setBrush(QColor(255, 255, 0, 100))
                    painter.drawEllipse(wx - 12, wy - 12, 24, 24)
                
                color = QColor(*track.color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color)
                painter.drawEllipse(wx - 5, wy - 5, 10, 10)
                
                if is_corrected:
                    painter.setPen(QPen(QColor(255, 0, 0), 2))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawEllipse(wx - 8, wy - 8, 16, 16)
                
                if self._show_track_ids:
                    painter.setPen(QColor(255, 255, 255))
                    painter.setFont(QFont("Microsoft YaHei", 9))
                    painter.drawText(wx + 8, wy - 8, str(track_id))

                marker_texts = []
                if track.suspicious_frames.get(self._current_frame_idx, False):
                    marker_texts.append("!")
                if track.keyframe_frames.get(self._current_frame_idx, False):
                    marker_texts.append("M")
                if marker_texts:
                    painter.setPen(self._marker_color)
                    painter.setFont(QFont("Microsoft YaHei", 22, QFont.Weight.Bold))
                    for idx, marker_text in enumerate(marker_texts):
                        painter.drawText(wx - 10 + idx * 18, wy - 20, marker_text)
        
        # 绘制消失中的追踪点
        if self._show_track_points:
            for track_id, pos in self._disappearing_positions.items():
                wx, wy = self._video_to_widget_coords(pos[0], pos[1])
                
                if track_id == self._selected_track_id:
                    painter.setPen(QPen(QColor(255, 255, 0), 3))
                    painter.setBrush(QColor(255, 255, 0, 100))
                    painter.drawEllipse(wx - 12, wy - 12, 24, 24)
                
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(*self._end_point_color))
                painter.drawEllipse(wx - 5, wy - 5, 10, 10)
                
                if self._show_track_ids:
                    painter.setPen(QColor(255, 255, 255))
                    painter.setFont(QFont("Microsoft YaHei", 9))
                    painter.drawText(wx + 8, wy - 8, str(track_id))
        
        # 绘制取色器参考圆
        if self._pick_color_mode and self._pick_color_circle_pos is not None:
            cx, cy = self._pick_color_circle_pos
            wx, wy = self._video_to_widget_coords(cx, cy)
            radius = int(self._pick_color_circle_radius * self._scale)
            
            # 绘制圆（黄色半透明）
            painter.setPen(QPen(QColor(255, 255, 0), 2))
            painter.setBrush(QColor(255, 255, 0, 50))
            painter.drawEllipse(wx - radius, wy - radius, radius * 2, radius * 2)
            
            # 绘制十字线
            painter.setPen(QPen(QColor(255, 0, 0), 1))
            painter.drawLine(wx - radius - 5, wy, wx + radius + 5, wy)
            painter.drawLine(wx, wy - radius - 5, wx, wy + radius + 5)
    
    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下事件"""
        pos = event.pos()
        
        if not self._is_inside_video_area(pos.x(), pos.y()):
            self.setFocus()
            return
        
        video_coords = self._widget_to_video_coords(pos.x(), pos.y())
        if video_coords is None:
            return
        vx, vy = video_coords
        
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            if event.button() == Qt.MouseButton.LeftButton:
                self._vertical_line_xs.append(float(vx))
                self._vertical_line_xs = sorted(self._vertical_line_xs)
                self.vertical_lines_changed.emit(self._vertical_line_xs.copy())
                self.update()
                return
            if event.button() == Qt.MouseButton.RightButton and self._vertical_line_xs:
                nearest_idx = min(range(len(self._vertical_line_xs)), key=lambda i: abs(self._vertical_line_xs[i] - vx))
                del self._vertical_line_xs[nearest_idx]
                self.vertical_lines_changed.emit(self._vertical_line_xs.copy())
                self.update()
                return
        
        # 取色器模式
        if self._pick_color_mode and event.button() == Qt.MouseButton.LeftButton:
            color = self._get_pixel_color(int(vx), int(vy))
            if color:
                self.color_picked.emit(int(vx), int(vy), color)
            return
        
        if event.button() == Qt.MouseButton.LeftButton:
            track_id = self._find_track_at_position(pos.x(), pos.y())
            footprint_id = self._find_footprint_at_position(pos.x(), pos.y())
            
            if track_id is not None:
                self._selected_footprint_id = None
                is_disappearing = track_id in self._disappearing_positions
                
                if is_disappearing:
                    dvx, dvy = self._disappearing_positions[track_id]
                    self.restore_track_requested.emit(track_id, self._current_frame_idx, dvx, dvy)
                    self._selected_track_id = track_id
                    self._dragged_track_id = track_id
                    self._is_dragging = True
                    self.track_selected.emit(track_id)
                elif self._end_mode_enabled and self._ctrl_pressed:
                    cvx, cvy = self._current_positions[track_id]
                    self.end_track_requested.emit(track_id, self._current_frame_idx, cvx, cvy)
                else:
                    self._selected_track_id = track_id
                    self._dragged_track_id = track_id
                    self._is_dragging = True
                    # 发送选中信号
                    from core.tracker import Track
                    self.track_selected.emit(track_id)
            elif footprint_id is not None:
                self._selected_track_id = None
                self._selected_footprint_id = footprint_id
                self._is_dragging = False
                self._dragged_track_id = None
            else:
                self._selected_track_id = None
                self._selected_footprint_id = None
                if (event.modifiers() & Qt.KeyboardModifier.AltModifier) and self._add_mode_enabled:
                    self.add_track_requested.emit(vx, vy, self._current_frame_idx)
                elif self._add_footprint_mode_enabled:
                    self.add_footprint_requested.emit(vx, vy, self._current_frame_idx)
            self.update()
        
        self.setFocus()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动事件"""
        pos = event.pos()
        
        # 取色器模式：更新参考圆位置
        if self._pick_color_mode:
            video_coords = self._widget_to_video_coords(pos.x(), pos.y())
            if video_coords:
                self._pick_color_circle_pos = (int(video_coords[0]), int(video_coords[1]))
                self.update()
            return
        
        if self._is_dragging and self._dragged_track_id is not None:
            video_coords = self._widget_to_video_coords(pos.x(), pos.y())
            if video_coords is None:
                return
            vx, vy = video_coords
            
            self.track_corrected.emit(self._dragged_track_id, self._current_frame_idx, vx, vy)
            
            if self._dragged_track_id in self._current_positions:
                self._current_positions[self._dragged_track_id] = (vx, vy)
            elif self._dragged_track_id in self._disappearing_positions:
                self._disappearing_positions[self._dragged_track_id] = (vx, vy)
            self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = False
            self._dragged_track_id = None
    
    def keyPressEvent(self, event: QKeyEvent):
        """键盘按下事件"""
        if event.key() == Qt.Key.Key_Control:
            self._ctrl_pressed = True
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif event.key() == Qt.Key.Key_Delete:
            if self._selected_track_id is not None and self._del_mode_enabled:
                self.remove_track_requested.emit(self._selected_track_id)
                self._selected_track_id = None
                self.update()
                return
            if self._selected_footprint_id is not None and self._del_footprint_mode_enabled:
                self.remove_footprint_requested.emit(self._selected_footprint_id)
                self._selected_footprint_id = None
                self.update()
                return
        super().keyPressEvent(event)
    
    def keyReleaseEvent(self, event: QKeyEvent):
        """键盘释放事件"""
        if event.key() == Qt.Key.Key_Control:
            self._ctrl_pressed = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().keyReleaseEvent(event)
    
    def resizeEvent(self, event):
        """尺寸变化事件"""
        super().resizeEvent(event)
        self.update()
    
    # 信号别名（兼容）
    track_selected = pyqtSignal(int)
