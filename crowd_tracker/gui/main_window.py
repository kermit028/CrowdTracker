"""
主窗口 - 整合所有组件
"""
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
    QFileDialog, QMessageBox, QSplitter, QDialog
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QKeySequence, QShortcut
import cv2
import numpy as np
from typing import Optional, Tuple, Dict

from gui.video_widget import VideoWidget
from gui.control_panel import ControlPanel
from gui.init_dialog import PerspectiveCorrectionDialog
from core.tracker import CrowdTracker, Track
from datetime import datetime
import os


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人群追踪器 - Crowd Tracker")
        self.setMinimumSize(1200, 700)
        
        # 视频相关
        self._video_path: Optional[str] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._total_frames = 0
        self._fps = 30.0
        self._current_frame_idx = 0
        self._frame_width = 0
        self._frame_height = 0
        
        # 追踪器
        self._tracker = CrowdTracker()
        self._is_tracking = False
        self._is_initialized = False
        
        # 编号自动排序
        self._auto_sort_ids = True
        
        # 取色器状态
        self._pick_color_mode = False
        
        # 色块追踪状态缓存
        self._last_on_block_status = {}
        
        # 播放控制
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_timer)
        self._play_speed = 1  # 播放速度倍数
        
        # 修正模式
        self._correction_mode = False
        self._selected_track_id: Optional[int] = None
        
        self._frame_correction_enabled = False
        self._frame_homography: Optional[np.ndarray] = None
        self._frame_correction_points = []
        self._footprint_points = []
        self._next_footprint_id = 1
        
        self._setup_ui()
    
    def _setup_ui(self):
        """设置UI"""
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 分割器
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(8)  # 增加分割条宽度，便于拖动
        main_layout.addWidget(self.splitter)
        
        # 设置全局快捷键
        self._setup_shortcuts()
        
        # 左侧：视频显示
        self.video_widget = VideoWidget()
        self.video_widget.setMinimumWidth(500)
        self.video_widget.track_corrected.connect(self._on_track_corrected)
        self.video_widget.add_track_requested.connect(self._on_add_track_point)
        self.video_widget.add_footprint_requested.connect(self._on_add_footprint_point)
        self.video_widget.track_selected.connect(self._on_track_selected)
        self.video_widget.remove_track_requested.connect(self._remove_track_point)
        self.video_widget.remove_footprint_requested.connect(self._remove_footprint_point)
        self.video_widget.end_track_requested.connect(self._on_end_track_requested)
        self.video_widget.restore_track_requested.connect(self._on_restore_track_requested)
        self.video_widget.color_picked.connect(self._on_video_color_picked)
        self.splitter.addWidget(self.video_widget)
        
        # 右侧：控制面板
        self.control_panel = ControlPanel()
        self.control_panel.setMinimumWidth(420)  # 增加最小宽度到420
        self.control_panel.setMaximumWidth(600)  # 设置最大宽度
        self.control_panel.load_video_clicked.connect(self._load_video)
        self.control_panel.load_trajectory_clicked.connect(self._load_trajectory)
        self.control_panel.play_clicked.connect(self._start_playback)
        self.control_panel.pause_clicked.connect(self._stop_playback)
        self.control_panel.next_frame_clicked.connect(self._next_frame)
        self.control_panel.prev_frame_clicked.connect(self._prev_frame)
        self.control_panel.slider_changed.connect(self._seek_frame)
        self.control_panel.goto_frame_clicked.connect(self._goto_frame)
        self.control_panel.save_results_clicked.connect(self._save_results)
        self.control_panel.clear_data_clicked.connect(self._clear_all_tracks_and_footprints)
        self.control_panel.show_points_changed.connect(self.video_widget.set_show_track_points)
        self.control_panel.show_trajectory_changed.connect(self.video_widget.set_show_trajectory)
        self.control_panel.show_ids_changed.connect(self.video_widget.set_show_track_ids)
        self.control_panel.show_footprints_changed.connect(self.video_widget.set_show_footprints)
        self.control_panel.trajectory_length_changed.connect(self.video_widget.set_trajectory_frame_count)
        self.control_panel.add_mode_changed.connect(self.video_widget.set_add_mode_enabled)
        self.control_panel.del_mode_changed.connect(self.video_widget.set_del_mode_enabled)
        self.control_panel.add_footprint_mode_changed.connect(self.video_widget.set_add_footprint_mode_enabled)
        self.control_panel.del_footprint_mode_changed.connect(self.video_widget.set_del_footprint_mode_enabled)
        self.control_panel.end_mode_changed.connect(self.video_widget.set_end_mode_enabled)
        self.control_panel.auto_sort_ids_changed.connect(self._on_auto_sort_changed)
        self.control_panel.pick_color_mode_changed.connect(self._on_pick_color_mode_changed)
        self.control_panel.correction_setup_clicked.connect(self._on_setup_frame_correction)
        self.control_panel.correction_clear_clicked.connect(self._on_clear_frame_correction)
        self.control_panel.correction_enabled_changed.connect(self._on_frame_correction_enabled_changed)
        self.control_panel.vertical_lines_enabled_changed.connect(self.video_widget.set_show_vertical_lines)
        self.control_panel.vertical_lines_list_changed.connect(self.video_widget.set_vertical_line_positions)
        self.control_panel.vertical_lines_clear_clicked.connect(self.video_widget.clear_vertical_lines)
        self.control_panel.track_note_changed.connect(self._on_track_note_changed)
        self.control_panel.footprint_track_changed.connect(self._on_footprint_track_changed)
        self.control_panel.keyframe_set_clicked.connect(self._on_keyframe_set_clicked)
        self.control_panel.keyframe_target_text_changed.connect(self._on_keyframe_target_text_changed)
        self.splitter.addWidget(self.control_panel)
        
        self.video_widget.vertical_lines_changed.connect(self.control_panel.set_vertical_lines_positions)
        self.video_widget.set_add_mode_enabled(self.control_panel.is_add_mode_enabled())
        self.video_widget.set_del_mode_enabled(self.control_panel.is_del_mode_enabled())
        self.video_widget.set_end_mode_enabled(self.control_panel.is_end_mode_enabled())
        self.video_widget.set_trajectory_frame_count(self.control_panel.get_trajectory_length_frames())
        
        # 设置分割器比例（左侧:右侧 = 2:1）
        self.splitter.setSizes([800, 420])
        
        # 状态栏
        self.statusBar().showMessage("就绪 - 请先选择视频文件")
    
    def _load_video(self):
        """加载视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", 
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
        
        # 释放旧的视频
        if self._cap is not None:
            self._cap.release()
        
        # 打开新视频
        self._cap = cv2.VideoCapture(file_path)
        if not self._cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开视频文件！")
            return
        
        self._video_path = file_path
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._current_frame_idx = 0
        
        self._frame_correction_enabled = False
        self._frame_homography = None
        self._frame_correction_points = []
        self._footprint_points = []
        self._next_footprint_id = 1
        self.video_widget.set_vertical_line_positions([])
        self.video_widget.set_footprints([])
        self.control_panel.set_vertical_lines_positions([])
        self.control_panel.set_keyframe_target_track_id(None)
        self.control_panel.set_keyframe_count(0)
        
        # 更新追踪器
        self._tracker = CrowdTracker(fps=int(self._fps))
        self.video_widget.set_fps(int(self._fps))
        
        # 更新控制面板
        self.control_panel.set_video_info(
            file_path, self._total_frames, self._fps, 
            self._frame_width, self._frame_height
        )
        
        # 读取第一帧
        ret, frame = self._cap.read()
        if ret:
            frame = self._apply_frame_correction(frame)
            self.video_widget.set_frame(frame, 0)
            self._update_display()
        
        self.control_panel.set_correction_available(self._frame_homography is not None, self._frame_correction_enabled)
        
        self._is_initialized = True
        self.statusBar().showMessage(f"视频已加载，Alt+左键添加追踪点，左键添加脚印点: {file_path}")
    
    def _on_add_track_point(self, x: float, y: float, frame_idx: int):
        """添加新追踪点"""
        if self._cap is None:
            return
        
        # 退出取色器模式
        if self._pick_color_mode:
            self.control_panel.set_pick_color_mode(False)
            self._pick_color_mode = False
        
        # 确保我们在正确的帧
        if frame_idx != self._current_frame_idx:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self._cap.read()
            if not ret:
                return
        else:
            # 使用当前帧
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self._cap.read()
            if not ret:
                return
        
        frame = self._apply_frame_correction(frame)
        self.video_widget.set_frame(frame, frame_idx)
        
        # 添加新追踪点（使用用户选择的颜色）
        color = self.control_panel.get_new_point_color()
        track_id = self._tracker.add_track_point(frame, frame_idx, (x, y), color=color)
        
        # 更新显示
        self._update_display()
        self.statusBar().showMessage(f"已添加追踪点 #{track_id} 在位置 ({x:.1f}, {y:.1f})")
    
    def _on_add_footprint_point(self, x: float, y: float, frame_idx: int):
        """添加脚印点"""
        related_track_id = self._auto_assign_footprint_track_id(x)
        self._footprint_points.append({
            "id": self._next_footprint_id,
            "frame_idx": frame_idx,
            "x": x,
            "y": y,
            "color": self.control_panel.get_new_footprint_color(),
            "track_id": related_track_id,
        })
        self._next_footprint_id += 1
        self._update_display()
        relate_text = f"，对应追踪点 #{related_track_id}" if related_track_id is not None else ""
        self.statusBar().showMessage(f"已添加脚印点 在帧 {frame_idx} 的位置 ({x:.1f}, {y:.1f}){relate_text}")
    
    def _start_playback(self):
        """开始播放"""
        if self._cap is None:
            return
        
        # 退出取色器模式
        if self._pick_color_mode:
            self.control_panel.set_pick_color_mode(False)
            self._pick_color_mode = False
        
        # 计算定时器间隔
        interval = int(1000 / (self._fps * self._play_speed))
        self._timer.start(max(1, interval))
        self._is_tracking = True
        self.statusBar().showMessage("播放中...")
    
    def _stop_playback(self):
        """停止播放"""
        self._timer.stop()
        self._is_tracking = False
        self.statusBar().showMessage("已暂停")
    
    def _next_frame(self):
        """下一帧"""
        if self._cap is None:
            return
        
        if self._current_frame_idx >= self._total_frames - 1:
            return
        
        # 退出取色器模式
        if self._pick_color_mode:
            self.control_panel.set_pick_color_mode(False)
            self._pick_color_mode = False
        
        self._current_frame_idx = min(self._total_frames - 1, self._current_frame_idx + 1)
        self._process_current_frame()
    
    def _prev_frame(self):
        """上一帧"""
        if self._cap is None:
            return
        
        if self._current_frame_idx <= 0:
            return
        
        # 退出取色器模式
        if self._pick_color_mode:
            self.control_panel.set_pick_color_mode(False)
            self._pick_color_mode = False
        
        self._current_frame_idx = max(0, self._current_frame_idx - 1)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._current_frame_idx)
        
        ret, frame = self._cap.read()
        if ret:
            frame = self._apply_frame_correction(frame)
            self.video_widget.set_frame(frame, self._current_frame_idx)
            
            # 如果有追踪点，需要重新初始化光流状态
            if self._tracker.tracks:
                # 清空消失状态
                self._tracker.clear_disappearing_tracks()
                
                # 重置上一帧灰度图，使光流从当前帧重新开始
                self._tracker._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 更新追踪点数组为当前帧的已知位置
                new_points = []
                new_ids = []
                for track_id, track in self._tracker.tracks.items():
                    pos = track.get_position_at(self._current_frame_idx)
                    if pos:
                        new_points.append([[pos[0], pos[1]]])
                        new_ids.append(track_id)
                
                if new_points:
                    self._tracker._track_points = np.array(new_points, dtype=np.float32)
                    self._tracker._active_track_ids = new_ids
                else:
                    self._tracker._track_points = None
                    self._tracker._active_track_ids = []
            
            # 如果启用了色块追踪，执行一次色块检测
            self._apply_color_tracking_if_enabled(frame)
            
            self._update_display()
    
    def _apply_color_tracking_if_enabled(self, frame):
        """如果启用了色块追踪，则执行一次"""
        if not self.control_panel.is_auto_tracking_enabled():
            self._last_on_block_status = {}
            return

        auto_add = self.control_panel.is_auto_add_enabled()
        auto_correct = self.control_panel.is_auto_correct_enabled()
        auto_end = self.control_panel.is_auto_end_enabled()
        
        # 清空上一次的色块状态
        self._last_on_block_status = {}
        
        if auto_add or auto_correct or auto_end:
            target_color = self.control_panel.get_block_color()
            tolerance = self.control_panel.get_color_tolerance()
            min_size = self.control_panel.get_min_pixel_size()
            
            result = self._tracker.process_color_tracking(
                frame, self._current_frame_idx,
                target_color, tolerance, min_size,
                auto_add, auto_correct, auto_end
            )
            
            # 保存色块匹配状态
            self._last_on_block_status = result.get('on_block_status', {})
            
            # 显示调试信息
            block_count = len(result.get('blocks', []))
            active_count = len(self._tracker._active_track_ids)
            matched_count = len(result.get('corrected', []))
            
            msgs = [f"检测到{block_count}个色块", f"{active_count}个追踪点"]
            if matched_count > 0:
                msgs.append(f"匹配{matched_count}个")
            self.statusBar().showMessage(f"色块追踪: {', '.join(msgs)}")
    
    def _seek_frame(self, frame_idx: int):
        """通过滑块跳转"""
        if self._cap is None:
            return
        
        # 退出取色器模式
        if self._pick_color_mode:
            self.control_panel.set_pick_color_mode(False)
            self._pick_color_mode = False
        
        self._current_frame_idx = frame_idx
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret, frame = self._cap.read()
        if ret:
            frame = self._apply_frame_correction(frame)
            self.video_widget.set_frame(frame, frame_idx)
            
            # 如果有追踪点，重新初始化光流状态
            if self._tracker.tracks:
                # 清空消失状态
                self._tracker.clear_disappearing_tracks()
                
                # 重置上一帧灰度图
                self._tracker._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 更新追踪点数组为当前帧的已知位置
                new_points = []
                new_ids = []
                for track_id, track in self._tracker.tracks.items():
                    pos = track.get_position_at(self._current_frame_idx)
                    if pos:
                        new_points.append([[pos[0], pos[1]]])
                        new_ids.append(track_id)
                
                if new_points:
                    self._tracker._track_points = np.array(new_points, dtype=np.float32)
                    self._tracker._active_track_ids = new_ids
                else:
                    self._tracker._track_points = None
                    self._tracker._active_track_ids = []
            
            # 如果启用了色块追踪，执行一次色块检测
            self._apply_color_tracking_if_enabled(frame)
            
            self._update_display()
    
    def _goto_frame(self, frame_idx: int):
        """跳转到指定帧"""
        if self._cap is None:
            return
        
        # 退出取色器模式
        if self._pick_color_mode:
            self.control_panel.set_pick_color_mode(False)
            self._pick_color_mode = False
        
        frame_idx = max(0, min(self._total_frames - 1, frame_idx))
        self._seek_frame(frame_idx)
    
    def _on_timer(self):
        """定时器回调"""
        if self._current_frame_idx >= self._total_frames - 1:
            self._stop_playback()
            return
        
        self._current_frame_idx += 1
        self._process_current_frame()
    
    def _process_current_frame(self):
        """处理当前帧"""
        if self._cap is None:
            return
        
        ret, frame = self._cap.read()
        if not ret:
            self._stop_playback()
            return
        
        frame = self._apply_frame_correction(frame)
        
        # 更新显示
        self.video_widget.set_frame(frame, self._current_frame_idx)
        
        # 色块自动追踪处理
        auto_tracking_enabled = self.control_panel.is_auto_tracking_enabled()
        auto_add = self.control_panel.is_auto_add_enabled()
        auto_correct = self.control_panel.is_auto_correct_enabled()
        auto_end = self.control_panel.is_auto_end_enabled()
        
        # 清空上一次的色块状态
        self._last_on_block_status = {}
        
        if auto_tracking_enabled and (auto_add or auto_correct or auto_end):
            # 获取色块追踪参数
            target_color = self.control_panel.get_block_color()
            tolerance = self.control_panel.get_color_tolerance()
            min_size = self.control_panel.get_min_pixel_size()
            
            # 执行色块追踪
            result = self._tracker.process_color_tracking(
                frame, self._current_frame_idx,
                target_color, tolerance, min_size,
                auto_add, auto_correct, auto_end
            )
            
            # 保存色块匹配状态
            self._last_on_block_status = result.get('on_block_status', {})
            
            # 构建调试信息
            block_count = len(result.get('blocks', []))
            active_count = len(self._tracker._active_track_ids)
            matched_count = len(result.get('corrected', []))
            
            msgs = []
            msgs.append(f"检测到{block_count}个色块")
            msgs.append(f"{active_count}个追踪点")
            if matched_count > 0:
                msgs.append(f"匹配{matched_count}个")
            if result['added']:
                msgs.append(f"新增{len(result['added'])}个")
            if result['ended']:
                msgs.append(f"结束{len(result['ended'])}个")
            
            self.statusBar().showMessage(f"色块追踪: {', '.join(msgs)}")
        
        # 如果有追踪点，处理帧（光流追踪）
        if auto_tracking_enabled and self._tracker.tracks:
            self._tracker.process_frame(frame, self._current_frame_idx)
        
        self._update_display()
    
    def _update_display(self):
        """更新显示"""
        # 更新控制面板
        self.control_panel.set_current_frame(self._current_frame_idx, self._fps)
        
        # 获取活跃的追踪点
        active_tracks = self._tracker.get_active_tracks(self._current_frame_idx)
        
        # 获取消失中的追踪点
        disappearing_tracks = self._tracker.get_disappearing_tracks()
        disappearing_positions = self._tracker.get_disappearing_positions()
        
        # 更新坐标列表（包含"在色块上"状态）
        total_tracks = len(self._tracker.tracks)
        self.control_panel.update_tracks_table(
            active_tracks, disappearing_tracks, total_tracks, 
            self._last_on_block_status, self._current_frame_idx
        )
        
        # 更新视频显示
        current_positions = {}
        for track_id, track in active_tracks.items():
            pos = track.get_position_at(self._current_frame_idx)
            if pos:
                current_positions[track_id] = pos
        
        visible_footprints = self._get_visible_footprints(self._current_frame_idx)
        self.video_widget.set_tracks(active_tracks, current_positions, disappearing_tracks, disappearing_positions)
        self.video_widget.set_footprints(visible_footprints)
        self.video_widget.set_end_point_color(self.control_panel.get_end_point_color())
        self.control_panel.update_footprints_table(visible_footprints)
        self._refresh_keyframe_target_summary()
    
    def _on_pick_color_mode_changed(self, enabled: bool):
        """取色器模式切换"""
        self._pick_color_mode = enabled
        self.video_widget.set_pick_color_mode(enabled)
        
        if enabled:
            # 获取像素尺寸并设置参考圆
            pixel_size = self.control_panel.get_min_pixel_size()
            self.video_widget.set_pick_color_circle(None, pixel_size // 2)
            self.statusBar().showMessage("取色模式：在视频画面上点击取色，显示像素尺寸参考圆")
        else:
            # 清除参考圆
            self.video_widget.set_pick_color_circle(None, 0)
            self.statusBar().showMessage("已退出取色模式")
    
    def _on_video_color_picked(self, x: int, y: int, color: tuple):
        """从视频画面取色"""
        # 设置基准颜色
        self.control_panel.set_block_color(color)
        # 保持取色模式，不自动退出，可以继续取色
        self.statusBar().showMessage(f"已取色: RGB{color} 位置: ({x}, {y}) - 继续点击取色或点击取色按钮退出")
    
    def _on_track_corrected(self, track_id: int, frame_idx: int, x: float, y: float):
        """追踪点被修正 - 拖拽时自动触发，无需按Ctrl
        
        修正后会清除该追踪点后续帧的轨迹，并从当前帧重新开始光流追踪
        """
        # 更新追踪器中的位置（clear_future=True 会清除后续轨迹）
        self._tracker.correct_track_position(track_id, frame_idx, (x, y), clear_future=True)
        
        # 重置光流状态，使后续帧从当前帧重新开始追踪
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self._cap.read()
            if ret:
                frame = self._apply_frame_correction(frame)
                # 重置追踪器的上一帧灰度图，使光流从当前帧重新开始
                self._tracker._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 更新显示
        self._update_display()
        
        if self.control_panel.is_auto_tracking_enabled():
            self.statusBar().showMessage(
                f"已修正追踪点 #{track_id} 在帧 {frame_idx} 的位置 ({x:.1f}, {y:.1f})，后续轨迹已清除，将从新位置重新跟踪"
            )
        else:
            self.statusBar().showMessage(
                f"已修正追踪点 #{track_id} 在帧 {frame_idx} 的位置 ({x:.1f}, {y:.1f})，后续轨迹已清除，可在开启自动跟踪后继续补轨迹"
            )
    
    def _remove_track_point(self, track_id: int):
        """移除追踪点"""
        # 退出取色器模式
        if self._pick_color_mode:
            self.control_panel.set_pick_color_mode(False)
            self._pick_color_mode = False
        
        self._tracker.remove_track_point(track_id)
        
        # 如果启用了编号自动排序，重新编号
        if self._auto_sort_ids:
            id_map = self._tracker.renumber_tracks()
            self._remap_footprint_track_ids(id_map)
            self.control_panel.remap_keyframe_target_ids(id_map)
        
        self._update_display()
        self.statusBar().showMessage(f"已移除追踪点 #{track_id}")
    
    def _on_end_track_requested(self, track_id: int, frame_idx: int, x: float, y: float):
        """结束追踪点"""
        self._tracker.end_track_point(track_id, frame_idx, (x, y))
        self._update_display()
        self.statusBar().showMessage(f"已结束追踪点 #{track_id}")
    
    def _on_restore_track_requested(self, track_id: int, frame_idx: int, x: float, y: float):
        """恢复消失中的追踪点"""
        self._tracker.restore_track_point(track_id, frame_idx, (x, y))
        self._update_display()
        self.statusBar().showMessage(f"已恢复追踪点 #{track_id}")
    
    def _on_track_selected(self, track_id: int):
        """追踪点被选中"""
        self._selected_track_id = track_id
        self.video_widget.set_selected_track(track_id)
        self.control_panel.set_keyframe_target_track_id(track_id)
        self.control_panel.set_keyframe_count(self._tracker.get_track_keyframe_count(track_id))
        self.statusBar().showMessage(f"选中追踪点 #{track_id}")
    
    def _on_track_note_changed(self, track_id: int, note: str):
        """更新追踪点备注名"""
        self._tracker.set_track_note(track_id, note)
        self.statusBar().showMessage(f"已更新追踪点 #{track_id} 的备注名：{note or '空'}")

    def _on_keyframe_set_clicked(self, track_ids):
        track_ids = [int(track_id) for track_id in track_ids if track_id is not None]
        if not track_ids:
            QMessageBox.information(self, "提示", "请先输入当前画面中的追踪点ID。")
            return

        success_ids = []
        failed_ids = []
        for track_id in track_ids:
            if self._tracker.set_track_keyframe(track_id, self._current_frame_idx, True):
                success_ids.append(track_id)
            else:
                failed_ids.append(track_id)

        self._update_display()
        if len(success_ids) == 1:
            self.control_panel.set_keyframe_target_track_id(success_ids[0])
            self.control_panel.set_keyframe_count(self._tracker.get_track_keyframe_count(success_ids[0]))
        else:
            self._refresh_keyframe_target_summary()

        messages = []
        if success_ids:
            messages.append(f"已将追踪点 #{', #'.join(str(track_id) for track_id in success_ids)} 在当前帧设为关键帧")
        if failed_ids:
            messages.append(f"以下ID不在当前画面中，未设置关键帧：{', '.join(str(track_id) for track_id in failed_ids)}")
        self.statusBar().showMessage("；".join(messages) if messages else "未设置关键帧")

    def _on_keyframe_target_text_changed(self, _text: str):
        self._refresh_keyframe_target_summary()

    def _refresh_keyframe_target_summary(self):
        ids = self.control_panel.get_keyframe_target_ids()
        if len(ids) == 1:
            self.control_panel.set_keyframe_count(self._tracker.get_track_keyframe_count(ids[0]))
        else:
            self.control_panel.set_keyframe_count(0)
    
    def _on_footprint_track_changed(self, footprint_id: int, track_id):
        """更新脚印点对应追踪点"""
        found = False
        normalized_track_id = None if track_id is None else int(track_id)
        for item in self._footprint_points:
            if item["id"] == footprint_id:
                item["track_id"] = normalized_track_id
                found = True
                break
        
        if found:
            self._update_display()
            if normalized_track_id is None:
                self.statusBar().showMessage(f"已清空脚印点 #{footprint_id} 的对应追踪点")
            else:
                self.statusBar().showMessage(f"已将脚印点 #{footprint_id} 关联到追踪点 #{normalized_track_id}")
    
    def _remove_footprint_point(self, footprint_id: int):
        """删除脚印点"""
        new_points = [item for item in self._footprint_points if item["id"] != footprint_id]
        if len(new_points) != len(self._footprint_points):
            self._footprint_points = new_points
            self._update_display()
            self.statusBar().showMessage(f"已删除脚印点 #{footprint_id}")
    
    def _load_trajectory(self):
        """加载轨迹、画面校正和竖线标记"""
        if self._cap is None:
            QMessageBox.warning(self, "警告", "请先加载视频文件！")
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择轨迹/校正文件", "",
            "Excel文件 (*.xlsx);;CSV文件 (*.csv);;文本文件 (*.txt);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
        
        if file_path.lower().endswith(".xlsx") and not self._is_likely_matching_session_filename(file_path):
            reply = QMessageBox.warning(
                self,
                "文件名提醒",
                "载入数据文件名与当前视频文件名看起来不对应。\n这可能不是当前视频对应的标注结果。\n\n是否仍然继续载入？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        
        try:
            success, metadata = self._tracker.import_session(file_path)
            if success:
                self._apply_session_metadata(metadata)
                
                if self._auto_sort_ids:
                    id_map = self._tracker.renumber_tracks()
                    self._remap_footprint_track_ids(id_map)
                    self.control_panel.remap_keyframe_target_ids(id_map)
                
                latest_frame_idx = 0
                if self._tracker.tracks:
                    latest_frame_idx = max(
                        frame_idx
                        for track in self._tracker.tracks.values()
                        for frame_idx, _ in track.positions
                    )
                self._current_frame_idx = max(0, min(self._total_frames - 1, latest_frame_idx))
                
                # 导入成功后，需要重新初始化光流状态
                # 获取当前帧用于初始化
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._current_frame_idx)
                ret, frame = self._cap.read()
                if ret:
                    frame = self._apply_frame_correction(frame)
                    self.video_widget.set_frame(frame, self._current_frame_idx)
                    # 初始化追踪器的灰度图，使光流能继续追踪
                    import numpy as np
                    self._tracker._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                self._update_display()
                track_count = len(self._tracker.tracks)
                has_extra = bool(self._frame_correction_points or self.video_widget.get_vertical_line_positions())
                self.statusBar().showMessage(f"已载入 {track_count} 条轨迹，并跳转到最新轨迹帧 {self._current_frame_idx}: {file_path}")
                QMessageBox.information(
                    self,
                    "成功",
                    f"已成功载入 {track_count} 条轨迹，当前已跳转到最新轨迹帧 {self._current_frame_idx}" + ("，并恢复画面校正/竖线标记" if has_extra else "")
                )
            else:
                QMessageBox.warning(self, "警告", "未能从文件中加载轨迹数据，文件可能为空或格式不正确")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"载入失败:\n{str(e)}")
    
    def _save_results(self):
        """保存追踪结果"""
        metadata = self._get_session_metadata()
        has_metadata = bool(
            metadata["frame_correction"]["points"] or
            metadata["vertical_lines"] or
            self._footprint_points
        )
        
        if not self._tracker.tracks and not has_metadata:
            QMessageBox.warning(self, "警告", "没有追踪数据或校正信息可保存！")
            return
        
        # 生成默认文件名：完整视频名 + 日期时间
        default_filename = ""
        if self._video_path:
            video_name = os.path.splitext(os.path.basename(self._video_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")  # 年月日时分秒
            default_filename = f"{video_name}{timestamp}.xlsx"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存追踪结果", default_filename,
            "Excel文件 (*.xlsx)"
        )
        
        if not file_path:
            return
        
        if not file_path.lower().endswith(".xlsx"):
            file_path += ".xlsx"
        
        try:
            export_metadata = dict(metadata)
            export_metadata["footprint_points"] = [
                {
                    "id": item["id"],
                    "frame_idx": item["frame_idx"],
                    "x": round(item["x"], 3),
                    "y": round(item["y"], 3),
                    "color": list(item["color"]),
                    "track_id": item.get("track_id"),
                }
                for item in self._footprint_points
            ]
            self._tracker.export_session(file_path, export_metadata)
            QMessageBox.information(self, "成功", f"追踪结果、脚印点、画面校正和竖线标记已保存到:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败:\n{str(e)}")

    def _clear_all_tracks_and_footprints(self):
        """清空所有轨迹和脚印，但保留画面校正与竖线标记"""
        if not self._tracker.tracks and not self._footprint_points:
            QMessageBox.information(self, "提示", "当前没有轨迹或脚印数据可清空。")
            return

        reply = QMessageBox.question(
            self,
            "确认清空",
            "将清空当前所有轨迹和脚印点。\n画面校正和竖线标记会保留。\n\n是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._tracker = CrowdTracker(fps=int(self._fps))
        self._footprint_points = []
        self._next_footprint_id = 1
        self._last_on_block_status = {}
        self.video_widget._selected_track_id = None
        self.video_widget._selected_footprint_id = None
        self.control_panel.set_keyframe_target_track_id(None)
        self.control_panel.set_keyframe_count(0)
        self.video_widget.set_tracks({}, {})
        self.video_widget.set_footprints([])
        self._update_display()
        self.statusBar().showMessage("已清空所有轨迹和脚印，已保留画面校正与竖线标记")
    
    def _on_auto_sort_changed(self, enabled: bool):
        """编号自动排序选项改变"""
        self._auto_sort_ids = enabled
        if enabled:
            # 如果启用，立即执行一次重新编号
            id_map = self._tracker.renumber_tracks()
            self._remap_footprint_track_ids(id_map)
            self.control_panel.remap_keyframe_target_ids(id_map)
            self._update_display()
            self.statusBar().showMessage("已启用编号自动排序，已重新编号所有追踪点")
    
    def keyPressEvent(self, event: QKeyEvent):
        """键盘事件"""
        if event.key() == Qt.Key.Key_Left:
            self._prev_frame()
        elif event.key() == Qt.Key.Key_Right:
            self._next_frame()
        elif event.key() == Qt.Key.Key_Control:
            self.video_widget._ctrl_pressed = True
            self.video_widget.setCursor(Qt.CursorShape.CrossCursor)
        elif event.key() == Qt.Key.Key_Space:
            if self._is_tracking:
                self._stop_playback()
            else:
                self._start_playback()
        else:
            super().keyPressEvent(event)
    
    def keyReleaseEvent(self, event: QKeyEvent):
        """键盘释放事件"""
        if event.key() == Qt.Key.Key_Control:
            self.video_widget._ctrl_pressed = False
            self.video_widget.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            super().keyReleaseEvent(event)
    
    def _setup_shortcuts(self):
        """设置全局快捷键"""
        # 左方向键 - 上一帧
        shortcut_left = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        shortcut_left.setContext(Qt.ShortcutContext.ApplicationShortcut)
        shortcut_left.activated.connect(self._prev_frame)
        
        # 右方向键 - 下一帧
        shortcut_right = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        shortcut_right.setContext(Qt.ShortcutContext.ApplicationShortcut)
        shortcut_right.activated.connect(self._next_frame)
        
        # 空格键 - 播放/暂停
        shortcut_space = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        shortcut_space.setContext(Qt.ShortcutContext.ApplicationShortcut)
        shortcut_space.activated.connect(self._toggle_playback)
    
    def _toggle_playback(self):
        """切换播放/暂停状态"""
        if self._is_tracking:
            self._stop_playback()
        else:
            self._start_playback()
    
    def closeEvent(self, event):
        """关闭事件"""
        if self._cap is not None:
            self._cap.release()
        event.accept()
    
    def _apply_frame_correction(self, frame: np.ndarray) -> np.ndarray:
        if not self._frame_correction_enabled or self._frame_homography is None:
            return frame
        
        h, w = frame.shape[:2]
        return cv2.warpPerspective(frame, self._frame_homography, (w, h))
    
    def _on_setup_frame_correction(self):
        if self._cap is None:
            return
        
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._current_frame_idx)
        ret, frame = self._cap.read()
        if not ret:
            return
        
        dlg = PerspectiveCorrectionDialog(frame, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        
        pts = dlg.get_selected_points()
        if len(pts) != 4:
            return
        
        self._frame_correction_points = [(float(x), float(y)) for x, y in pts]
        self._rebuild_frame_correction_homography()
        self._frame_correction_enabled = self._frame_homography is not None
        
        had_tracks = bool(self._tracker.tracks)
        if had_tracks:
            self._tracker = CrowdTracker(fps=int(self._fps))
            self._last_on_block_status = {}
        
        self.control_panel.set_correction_available(True, True)
        
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._current_frame_idx)
        ret, frame = self._cap.read()
        if ret:
            frame = self._apply_frame_correction(frame)
            self.video_widget.set_frame(frame, self._current_frame_idx)
            self._update_display()
        
        self.statusBar().showMessage("已设置画面校正，追踪数据已重置（坐标基于校正后的画面）" if had_tracks else "已设置画面校正：当前追踪坐标将基于校正后的画面")
    
    def _on_clear_frame_correction(self):
        if self._frame_homography is None and not self._frame_correction_enabled:
            return
        
        self._frame_correction_enabled = False
        self._frame_homography = None
        self._frame_correction_points = []
        
        had_tracks = bool(self._tracker.tracks)
        if had_tracks:
            self._tracker = CrowdTracker(fps=int(self._fps))
            self._last_on_block_status = {}
        
        self.control_panel.set_correction_available(False, False)
        
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._current_frame_idx)
            ret, frame = self._cap.read()
            if ret:
                self.video_widget.set_frame(frame, self._current_frame_idx)
                self._update_display()
        
        self.statusBar().showMessage("已清除画面校正，追踪数据已重置" if had_tracks else "已清除画面校正")
    
    def _on_frame_correction_enabled_changed(self, enabled: bool):
        if self._frame_homography is None:
            self._frame_correction_enabled = False
            self.control_panel.set_correction_available(False, False)
            return
        
        if enabled == self._frame_correction_enabled:
            return
        
        self._frame_correction_enabled = enabled
        
        had_tracks = bool(self._tracker.tracks)
        if had_tracks:
            self._tracker = CrowdTracker(fps=int(self._fps))
            self._last_on_block_status = {}
        
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._current_frame_idx)
            ret, frame = self._cap.read()
            if ret:
                frame = self._apply_frame_correction(frame)
                self.video_widget.set_frame(frame, self._current_frame_idx)
                self._update_display()
        
        self.statusBar().showMessage("已切换画面校正开关（追踪数据已重置）" if had_tracks else "已切换画面校正开关")
    
    @staticmethod
    def _order_quad_points(pts: np.ndarray) -> np.ndarray:
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(-1)
        
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _rebuild_frame_correction_homography(self):
        if len(self._frame_correction_points) != 4 or self._frame_width <= 0 or self._frame_height <= 0:
            self._frame_homography = None
            return
        
        src = np.array(self._frame_correction_points, dtype=np.float32)
        src = self._order_quad_points(src)
        dst = np.array(
            [[0, 0], [self._frame_width - 1, 0], [self._frame_width - 1, self._frame_height - 1], [0, self._frame_height - 1]],
            dtype=np.float32
        )
        self._frame_homography = cv2.getPerspectiveTransform(src, dst)

    def _get_session_metadata(self) -> dict:
        return {
            "version": 1,
            "frame_correction": {
                "enabled": bool(self._frame_correction_enabled and self._frame_homography is not None),
                "points": [[round(x, 3), round(y, 3)] for x, y in self._frame_correction_points],
            },
            "vertical_lines": [round(x, 3) for x in sorted(self.video_widget.get_vertical_line_positions())],
            "display": {
                "trajectory_length_frames": self.control_panel.get_trajectory_length_frames(),
            },
        }

    def _apply_session_metadata(self, metadata: dict):
        display = metadata.get("display") or {}
        trajectory_length_frames = max(1, int(display.get("trajectory_length_frames", 10)))
        self.control_panel.set_trajectory_length_frames(trajectory_length_frames)
        self.video_widget.set_trajectory_frame_count(trajectory_length_frames)

        correction = metadata.get("frame_correction") or {}
        points = correction.get("points") or []
        
        if len(points) == 4:
            self._frame_correction_points = [(float(p[0]), float(p[1])) for p in points]
            self._rebuild_frame_correction_homography()
            self._frame_correction_enabled = bool(correction.get("enabled", True) and self._frame_homography is not None)
            self.control_panel.set_correction_available(self._frame_homography is not None, self._frame_correction_enabled)
        else:
            self._frame_correction_points = []
            self._frame_homography = None
            self._frame_correction_enabled = False
            self.control_panel.set_correction_available(False, False)
        
        vertical_lines = [float(x) for x in metadata.get("vertical_lines", [])]
        self.video_widget.set_vertical_line_positions(sorted(vertical_lines))
        self.control_panel.set_vertical_lines_positions(sorted(vertical_lines))
        
        footprint_points = []
        raw_footprints = metadata.get("footprint_points") or []
        if isinstance(raw_footprints, dict):
            for frame_idx, points in raw_footprints.items():
                try:
                    frame_no = int(frame_idx)
                    for x, y in points:
                        footprint_points.append({
                            "id": len(footprint_points) + 1,
                            "frame_idx": frame_no,
                            "x": float(x),
                            "y": float(y),
                            "color": self.control_panel.get_new_footprint_color(),
                            "track_id": self._auto_assign_footprint_track_id(float(x)),
                        })
                except (TypeError, ValueError):
                    continue
        else:
            for item in raw_footprints:
                try:
                    color = tuple(item.get("color", self.control_panel.get_new_footprint_color()))
                    footprint_points.append({
                        "id": int(item.get("id", len(footprint_points) + 1)),
                        "frame_idx": int(item["frame_idx"]),
                        "x": float(item["x"]),
                        "y": float(item["y"]),
                        "color": (int(color[0]), int(color[1]), int(color[2])),
                        "track_id": int(item["track_id"]) if item.get("track_id") is not None else None,
                    })
                except (TypeError, ValueError, KeyError, IndexError):
                    continue
        self._footprint_points = sorted(footprint_points, key=lambda item: (item["frame_idx"], item["id"]))
        self._next_footprint_id = max((item["id"] for item in self._footprint_points), default=0) + 1

    def _get_visible_footprints(self, frame_idx: int) -> list:
        if not self._footprint_points:
            return []
        
        max_duration_frames = max(1, int(round(self._fps * 0.5)))
        ordered = sorted(self._footprint_points, key=lambda item: (item["frame_idx"], item["id"]))
        visible = []
        
        for idx, item in enumerate(ordered):
            next_frame_idx = None
            for later in ordered[idx + 1:]:
                if later.get("track_id") == item.get("track_id") and item.get("track_id") is not None:
                    next_frame_idx = later["frame_idx"]
                    break
            end_frame_idx = item["frame_idx"] + max_duration_frames
            if next_frame_idx is not None:
                end_frame_idx = min(end_frame_idx, next_frame_idx - 1)
            
            if item["frame_idx"] <= frame_idx <= end_frame_idx:
                visible.append(item)
        
        return visible

    def _auto_assign_footprint_track_id(self, footprint_x: float):
        if not self._tracker.tracks:
            return None
        
        candidates = []
        for track_id, track in self._tracker.get_active_tracks(self._current_frame_idx).items():
            pos = track.get_position_at(self._current_frame_idx)
            if pos is not None:
                candidates.append((track_id, pos[0]))
        
        if not candidates:
            return None
        
        return min(candidates, key=lambda item: abs(item[1] - footprint_x))[0]

    def _remap_footprint_track_ids(self, id_map: dict):
        if not id_map:
            return
        for item in self._footprint_points:
            track_id = item.get("track_id")
            if track_id is not None and track_id in id_map:
                item["track_id"] = id_map[track_id]

    def _is_likely_matching_session_filename(self, session_path: str) -> bool:
        if not self._video_path:
            return True
        
        video_name = os.path.splitext(os.path.basename(self._video_path))[0]
        session_name = os.path.splitext(os.path.basename(session_path))[0]
        
        session_prefix = session_name
        if len(session_name) > 14 and session_name[-14:].isdigit():
            session_prefix = session_name[:-14]
        
        return session_prefix == video_name or session_name.startswith(video_name)
