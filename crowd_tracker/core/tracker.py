"""
核心追踪引擎 - 封装光流追踪逻辑
"""
import csv
import json
import zipfile
import cv2
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape


@dataclass
class Track:
    """单个追踪点的轨迹信息"""
    track_id: int
    positions: List[Tuple[int, Tuple[float, float]]] = field(default_factory=list)  # [(frame_idx, (x, y)), ...]
    is_manual_corrected: Dict[int, bool] = field(default_factory=dict)  # frame_idx -> bool
    color: Tuple[int, int, int] = field(default_factory=lambda: (0, 255, 0))  # 追踪点显示颜色
    block_color: Optional[Tuple[int, int, int]] = field(default=None)  # 关联色块的颜色（用于自动追踪）
    note: str = ""  # 备注名
    start_frame: int = 0  # 首次出现帧
    suspicious_frames: Dict[int, bool] = field(default_factory=dict)  # frame_idx -> bool
    keyframe_frames: Dict[int, bool] = field(default_factory=dict)  # frame_idx -> bool
    
    def get_position_at(self, frame_idx: int) -> Optional[Tuple[float, float]]:
        """获取指定帧的位置"""
        for idx, pos in self.positions:
            if idx == frame_idx:
                return pos
        return None
    
    def get_recent_positions(self, current_frame: int, frame_count: int = 10) -> List[Tuple[int, Tuple[float, float]]]:
        """获取最近若干帧的位置（用于显示轨迹）"""
        frame_range = max(1, int(frame_count))
        start_frame = max(0, current_frame - frame_range + 1)
        filtered = [(idx, pos) for idx, pos in self.positions if start_frame <= idx <= current_frame]
        latest_by_frame = {}
        for idx, pos in filtered:
            latest_by_frame[idx] = pos
        return sorted(latest_by_frame.items(), key=lambda item: item[0])
    
    def correct_position(self, frame_idx: int, new_pos: Tuple[float, float], clear_future: bool = True):
        """修正指定帧的位置
        
        Args:
            frame_idx: 要修正的帧索引
            new_pos: 新的位置 (x, y)
            clear_future: 是否清除后续帧的轨迹（默认True，用于手动校正后重新跟踪）
        """
        # 找到并更新位置
        found = False
        for i, (idx, pos) in enumerate(self.positions):
            if idx == frame_idx:
                self.positions[i] = (idx, new_pos)
                self.is_manual_corrected[frame_idx] = True
                found = True
                # 记录找到的位置索引，后续可能需要删除后面的轨迹
                insert_idx = i
                break
        
        if not found:
            # 如果没有找到，添加新位置
            self.positions.append((frame_idx, new_pos))
            self.positions.sort(key=lambda x: x[0])
            self.is_manual_corrected[frame_idx] = True
            # 找到插入后的索引
            for i, (idx, pos) in enumerate(self.positions):
                if idx == frame_idx:
                    insert_idx = i
                    break

        if frame_idx in self.suspicious_frames:
            del self.suspicious_frames[frame_idx]
        
        # 如果要求清除后续轨迹，删除当前帧之后的所有位置记录
        if clear_future:
            # 只保留当前帧及之前的位置
            self.positions = self.positions[:insert_idx + 1]
            # 清除后续帧的手动校正标记
            keys_to_remove = [k for k in self.is_manual_corrected.keys() if k > frame_idx]
            for k in keys_to_remove:
                del self.is_manual_corrected[k]
            suspicious_to_remove = [k for k in self.suspicious_frames.keys() if k > frame_idx]
            for k in suspicious_to_remove:
                del self.suspicious_frames[k]
            keyframes_to_remove = [k for k in self.keyframe_frames.keys() if k > frame_idx]
            for k in keyframes_to_remove:
                del self.keyframe_frames[k]
    
    def get_last_position_before(self, frame_idx: int) -> Optional[Tuple[int, Tuple[float, float]]]:
        """获取指定帧之前最后一个已知位置"""
        candidates = [(idx, pos) for idx, pos in self.positions if idx < frame_idx]
        if candidates:
            return candidates[-1]
        return None

    def get_recent_motion_positions(self, frame_idx: int, max_count: int = 6) -> List[Tuple[int, Tuple[float, float]]]:
        candidates = [(idx, pos) for idx, pos in self.positions if idx < frame_idx]
        return candidates[-max_count:]


class CrowdTracker:
    """人群追踪器 - 基于光流法"""
    
    def __init__(self, fps: int = 30):
        self.fps = fps
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        
        # Lucas-Kanade 光流参数
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # 内部状态
        self._prev_gray: Optional[np.ndarray] = None
        self._current_frame_idx: int = -1
        self._track_points: Optional[np.ndarray] = None  # 当前追踪点数组，用于光流计算
        self._active_track_ids: List[int] = []  # 当前活跃的点对应的track_id
        self._disappearing_tracks: Dict[int, Tuple[float, float]] = {}  # 正在消失的追踪点 -> 最后位置

    def _estimate_motion_constraint(self, track: Track, frame_idx: int) -> Tuple[np.ndarray, float]:
        history = track.get_recent_motion_positions(frame_idx, max_count=6)
        if not history:
            return np.array([0.0, 0.0], dtype=np.float32), 25.0
        if len(history) == 1:
            last_pos = np.array(history[-1][1], dtype=np.float32)
            return last_pos, 25.0

        positions = [np.array(pos, dtype=np.float32) for _, pos in history]
        last_pos = positions[-1]
        prev_pos = positions[-2]
        velocity = last_pos - prev_pos
        predicted = last_pos + velocity

        step_distances = [
            float(np.linalg.norm(positions[i] - positions[i - 1]))
            for i in range(1, len(positions))
        ]
        median_step = float(np.median(step_distances)) if step_distances else 0.0
        max_step = max(step_distances) if step_distances else 0.0
        last_step = float(np.linalg.norm(velocity))

        # 轨迹通常连续，这里根据历史运动幅度动态估计下一帧允许的最大偏移
        allowed = max(20.0, median_step * 3.5, max_step * 2.2, last_step * 2.5)
        return predicted, allowed

    def _apply_motion_constraint(self, track: Track, frame_idx: int, detected_pos: np.ndarray) -> Tuple[np.ndarray, bool]:
        predicted, allowed = self._estimate_motion_constraint(track, frame_idx)
        delta = detected_pos - predicted
        distance = float(np.linalg.norm(delta))

        if distance <= allowed:
            track.suspicious_frames.pop(frame_idx, None)
            return detected_pos, False

        if distance > 1e-6:
            constrained = predicted + delta * (allowed / distance)
        else:
            constrained = predicted

        track.suspicious_frames[frame_idx] = True
        return constrained.astype(np.float32), True
        
    def initialize_tracks(self, frame: np.ndarray, points: List[Tuple[float, float]]):
        """初始化追踪点"""
        if len(points) == 0:
            return
            
        self._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._current_frame_idx = 0
        
        # 创建追踪点数组
        self._track_points = np.array([[p] for p in points], dtype=np.float32)
        
        # 为每个点创建 Track 对象
        self._active_track_ids = []
        for i, point in enumerate(points):
            track_id = self.next_track_id
            self.next_track_id += 1
            
            color = tuple(np.random.randint(0, 255, 3).tolist())
            track = Track(
                track_id=track_id,
                positions=[(0, point)],
                color=color,
                start_frame=0
            )
            self.tracks[track_id] = track
            self._active_track_ids.append(track_id)
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Dict[int, Tuple[float, float]]:
        """
        处理一帧图像，返回当前帧所有追踪点的位置
        """
        # 清空上一帧的消失点（它们已经显示过一帧绿色了）
        self._disappearing_tracks.clear()

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._current_frame_idx = frame_idx

        current_positions = {}

        # 当前帧如果已经有轨迹数据，则直接采用该数据，不再用光流覆盖它。
        stored_positions = {}
        for track_id, track in self.tracks.items():
            pos = track.get_position_at(frame_idx)
            if pos is not None:
                track.suspicious_frames.pop(frame_idx, None)
                stored_positions[track_id] = pos

        if self._prev_gray is None:
            if stored_positions:
                ordered_ids = sorted(stored_positions.keys())
                self._track_points = np.array(
                    [[[stored_positions[track_id][0], stored_positions[track_id][1]]] for track_id in ordered_ids],
                    dtype=np.float32
                )
                self._active_track_ids = ordered_ids
                current_positions.update(stored_positions)
            else:
                self._track_points = None
                self._active_track_ids = []
            self._prev_gray = frame_gray.copy()
            return current_positions

        good_new = []
        good_ids = []
        tracked_indices = []
        optical_flow_input = []

        if self._track_points is not None and len(self._track_points) > 0:
            for i, track_id in enumerate(self._active_track_ids):
                stored_pos = stored_positions.get(track_id)
                if stored_pos is not None:
                    current_positions[track_id] = stored_pos
                    good_new.append([[stored_pos[0], stored_pos[1]]])
                    good_ids.append(track_id)
                    continue

                tracked_indices.append(i)
                optical_flow_input.append(self._track_points[i])

        new_points = None
        status = None
        if optical_flow_input:
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray,
                frame_gray,
                np.array(optical_flow_input, dtype=np.float32),
                None,
                **self.lk_params
            )

        if new_points is not None and status is not None:
            for result_idx, (new, st) in enumerate(zip(new_points, status)):
                source_idx = tracked_indices[result_idx]
                track_id = self._active_track_ids[source_idx]
                track = self.tracks[track_id]

                if st == 1:
                    pos, _ = self._apply_motion_constraint(track, frame_idx, new.ravel().astype(np.float32))
                    pos_tuple = (float(pos[0]), float(pos[1]))
                    current_positions[track_id] = pos_tuple
                    good_new.append([[pos_tuple[0], pos_tuple[1]]])
                    good_ids.append(track_id)
                    track.positions.append((frame_idx, pos_tuple))
                else:
                    old_pos = self._track_points[source_idx].ravel()
                    self._disappearing_tracks[track_id] = (float(old_pos[0]), float(old_pos[1]))

        active_id_set = set(good_ids)
        for track_id in sorted(stored_positions.keys()):
            if track_id in active_id_set:
                continue
            stored_pos = stored_positions[track_id]
            current_positions[track_id] = stored_pos
            good_new.append([[stored_pos[0], stored_pos[1]]])
            good_ids.append(track_id)

        if good_new:
            self._track_points = np.array(good_new, dtype=np.float32)
            self._active_track_ids = good_ids
        else:
            self._track_points = None
            self._active_track_ids = []
        
        self._prev_gray = frame_gray.copy()
        return current_positions
    
    def correct_track_position(self, track_id: int, frame_idx: int, new_pos: Tuple[float, float], 
                               clear_future: bool = True):
        """修正指定追踪点在指定帧的位置
        
        Args:
            track_id: 追踪点ID
            frame_idx: 帧索引
            new_pos: 新位置 (x, y)
            clear_future: 是否清除后续帧的轨迹（默认True，用于手动校正）
        """
        if track_id not in self.tracks:
            return
            
        track = self.tracks[track_id]
        track.correct_position(frame_idx, new_pos, clear_future=clear_future)
        
        # 如果修正的是消失中的点，恢复为活跃状态
        if track_id in self._disappearing_tracks:
            del self._disappearing_tracks[track_id]
            
            # 重新加入追踪点数组
            new_point = np.array([[new_pos]], dtype=np.float32)
            if self._track_points is None or len(self._track_points) == 0:
                self._track_points = new_point
                self._active_track_ids = [track_id]
            else:
                self._track_points = np.vstack([self._track_points, new_point])
                self._active_track_ids.append(track_id)
        elif self._track_points is not None:
            # 如果修正的是当前帧或之前的帧，更新或添加追踪点
            if track_id in self._active_track_ids:
                # 更新现有追踪点位置
                for i, tid in enumerate(self._active_track_ids):
                    if tid == track_id:
                        self._track_points[i] = [[new_pos[0], new_pos[1]]]
                        break
            else:
                # 如果追踪点不在活跃列表中（可能在消失列表后被校正），重新添加
                new_point = np.array([[new_pos]], dtype=np.float32)
                if len(self._track_points) == 0:
                    self._track_points = new_point
                else:
                    self._track_points = np.vstack([self._track_points, new_point])
                self._active_track_ids.append(track_id)
                # 从消失列表中移除（如果存在）
                if track_id in self._disappearing_tracks:
                    del self._disappearing_tracks[track_id]
    
    def get_track_at_position(self, pos: Tuple[float, float], radius: float = 10) -> Optional[int]:
        """查找指定位置附近的追踪点"""
        if self._track_points is None:
            return None
        
        for i, track_id in enumerate(self._active_track_ids):
            point = self._track_points[i].ravel()
            dist = np.sqrt((point[0] - pos[0])**2 + (point[1] - pos[1])**2)
            if dist < radius:
                return track_id
        return None
    
    def get_active_tracks(self, frame_idx: int) -> Dict[int, Track]:
        """获取当前帧活跃的追踪点"""
        result = {}
        for track_id, track in self.tracks.items():
            if track.get_position_at(frame_idx) is not None:
                result[track_id] = track
        return result
    
    def add_track_point(self, frame: np.ndarray, frame_idx: int, pos: Tuple[float, float], color: Optional[Tuple[int, int, int]] = None) -> int:
        """
        在指定帧添加新的追踪点
        返回新追踪点的ID
        """
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 创建新的Track
        track_id = self.next_track_id
        self.next_track_id += 1
        
        if color is None:
            color = tuple(np.random.randint(0, 255, 3).tolist())
        track = Track(
            track_id=track_id,
            positions=[(frame_idx, pos)],
            color=color,
            start_frame=frame_idx
        )
        self.tracks[track_id] = track
        
        # 如果这是第一个点，初始化追踪状态
        if self._prev_gray is None:
            self._prev_gray = frame_gray
            self._current_frame_idx = frame_idx
        
        # 添加到追踪点数组
        new_point = np.array([[pos]], dtype=np.float32)
        if self._track_points is None or len(self._track_points) == 0:
            self._track_points = new_point
            self._active_track_ids = [track_id]
        else:
            self._track_points = np.vstack([self._track_points, new_point])
            self._active_track_ids.append(track_id)
        
        return track_id
    
    def remove_track_point(self, track_id: int):
        """移除指定的追踪点"""
        if track_id not in self.tracks:
            return
        
        # 从tracks字典中移除
        del self.tracks[track_id]
        
        # 从活跃列表中移除
        if track_id in self._active_track_ids:
            idx = self._active_track_ids.index(track_id)
            self._active_track_ids.pop(idx)
            
            # 从追踪点数组中移除
            if self._track_points is not None and len(self._track_points) > idx:
                self._track_points = np.delete(self._track_points, idx, axis=0)
                if len(self._track_points) == 0:
                    self._track_points = None
        
        # 从消失列表中移除
        if track_id in self._disappearing_tracks:
            del self._disappearing_tracks[track_id]
    
    def end_track_point(self, track_id: int, frame_idx: int, pos: Tuple[float, float]):
        """将活跃追踪点标记为结束（消失中）"""
        if track_id not in self.tracks:
            return
        
        # 从活跃列表中移除
        if track_id in self._active_track_ids:
            idx = self._active_track_ids.index(track_id)
            self._active_track_ids.pop(idx)
            
            if self._track_points is not None and len(self._track_points) > idx:
                self._track_points = np.delete(self._track_points, idx, axis=0)
                if len(self._track_points) == 0:
                    self._track_points = None
        
        # 加入消失列表
        self._disappearing_tracks[track_id] = pos
    
    def restore_track_point(self, track_id: int, frame_idx: int, pos: Tuple[float, float]):
        """恢复消失中的追踪点为活跃状态"""
        if track_id not in self.tracks:
            return
        
        # 从消失列表中移除
        if track_id in self._disappearing_tracks:
            del self._disappearing_tracks[track_id]
        
        # 重新加入活跃列表
        new_point = np.array([[pos]], dtype=np.float32)
        if self._track_points is None or len(self._track_points) == 0:
            self._track_points = new_point
            self._active_track_ids = [track_id]
        else:
            self._track_points = np.vstack([self._track_points, new_point])
            self._active_track_ids.append(track_id)
        
        # 记录位置并标记为手动修正
        track = self.tracks[track_id]
        track.correct_position(frame_idx, pos)
    
    def get_disappearing_tracks(self) -> Dict[int, Track]:
        """获取正在消失的追踪点"""
        result = {}
        for track_id in self._disappearing_tracks.keys():
            if track_id in self.tracks:
                result[track_id] = self.tracks[track_id]
        return result
    
    def get_disappearing_positions(self) -> Dict[int, Tuple[float, float]]:
        """获取正在消失的追踪点位置"""
        return self._disappearing_tracks.copy()
    
    def clear_disappearing_tracks(self):
        """清空正在消失的追踪点"""
        self._disappearing_tracks.clear()
    
    def export_trajectories(self, filepath: str):
        """导出轨迹数据到文件"""
        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "frame_idx", "x", "y", "is_corrected", "color_r", "color_g", "color_b", "note", "start_frame", "is_keyframe"])
            for track_id, track in sorted(self.tracks.items()):
                r, g, b = track.color
                for frame_idx, pos in track.positions:
                    is_corrected = track.is_manual_corrected.get(frame_idx, False)
                    is_keyframe = track.keyframe_frames.get(frame_idx, False)
                    writer.writerow([track_id, frame_idx, f"{pos[0]:.2f}", f"{pos[1]:.2f}", int(is_corrected), r, g, b, track.note, track.start_frame, int(is_keyframe)])
    
    def import_trajectories(self, filepath: str) -> bool:
        """从文件导入轨迹数据
        
        返回: 是否成功导入
        """
        try:
            with open(filepath, 'r', newline='', encoding='utf-8-sig') as f:
                rows = list(csv.reader(f))
            
            imported_tracks = self._parse_trajectory_rows(rows)
            if not imported_tracks:
                return False
            
            self._set_imported_tracks(imported_tracks)
            return True
            
        except Exception:
            return False
    
    def renumber_tracks(self):
        """重新编号所有追踪点，使其从1开始连续"""
        if not self.tracks:
            return {}
        
        # 创建新的连续编号映射
        old_to_new = {}
        new_tracks = {}
        
        for new_id, (old_id, track) in enumerate(sorted(self.tracks.items()), start=1):
            old_to_new[old_id] = new_id
            
            # 创建新的 Track 对象，保持所有数据但更新ID
            new_track = Track(
                track_id=new_id,
                positions=track.positions.copy(),
                is_manual_corrected=track.is_manual_corrected.copy(),
                color=track.color,
                note=track.note,
                start_frame=track.start_frame,
                suspicious_frames=track.suspicious_frames.copy(),
                keyframe_frames=track.keyframe_frames.copy()
            )
            new_tracks[new_id] = new_track
        
        # 更新追踪点数组中的ID
        if self._active_track_ids:
            self._active_track_ids = [
                old_to_new.get(tid, tid) for tid in self._active_track_ids
            ]
        
        # 更新消失列表中的ID
        if self._disappearing_tracks:
            new_disappearing = {}
            for old_id, pos in self._disappearing_tracks.items():
                new_id = old_to_new.get(old_id, old_id)
                new_disappearing[new_id] = pos
            self._disappearing_tracks = new_disappearing
        
        # 替换 tracks
        self.tracks = new_tracks
        self.next_track_id = len(new_tracks) + 1
        return old_to_new

    def export_session(self, filepath: str, metadata: Optional[dict] = None):
        """导出轨迹与附加元数据"""
        metadata = metadata or {}
        if filepath.lower().endswith('.xlsx'):
            self._export_session_xlsx(filepath, metadata)
        else:
            self.export_trajectories(filepath)

    def import_session(self, filepath: str) -> Tuple[bool, dict]:
        """导入轨迹与附加元数据，兼容旧版CSV/TXT"""
        try:
            if filepath.lower().endswith('.xlsx'):
                imported_tracks, metadata = self._import_session_xlsx(filepath)
                if not imported_tracks:
                    return False, {}
                self._set_imported_tracks(imported_tracks)
                return True, metadata
            
            success = self.import_trajectories(filepath)
            return success, {}
        except Exception:
            return False, {}

    def _set_imported_tracks(self, imported_tracks: Dict[int, Track]):
        for track in imported_tracks.values():
            track.positions.sort(key=lambda p: p[0])
        
        self.tracks = imported_tracks
        self.next_track_id = (max(imported_tracks.keys()) + 1) if imported_tracks else 1
        
        if self.tracks:
            max_frame = max(
                max((p[0] for p in t.positions), default=0)
                for t in self.tracks.values()
            )
            
            points = []
            active_ids = []
            for track_id, track in sorted(self.tracks.items()):
                pos = track.get_position_at(max_frame)
                if pos is None and track.positions:
                    _, pos = track.positions[-1]
                
                if pos:
                    points.append([[pos[0], pos[1]]])
                    active_ids.append(track_id)
            
            if points:
                self._track_points = np.array(points, dtype=np.float32)
                self._active_track_ids = active_ids
            else:
                self._track_points = None
                self._active_track_ids = []
        else:
            self._track_points = None
            self._active_track_ids = []
        
        self._disappearing_tracks.clear()

    def _parse_trajectory_rows(self, rows: List[List[str]]) -> Dict[int, Track]:
        imported_tracks: Dict[int, Track] = {}
        if not rows:
            return imported_tracks
        
        start_row = 0
        has_color = False
        note_index = -1
        start_frame_index = -1
        keyframe_index = -1
        if rows and rows[0]:
            first_cell = (rows[0][0] or '').strip().lstrip('\ufeff')
            if first_cell == 'track_id':
                header = [cell.strip() for cell in rows[0]]
                has_color = 'color_r' in header
                note_index = header.index('note') if 'note' in header else -1
                start_frame_index = header.index('start_frame') if 'start_frame' in header else -1
                keyframe_index = header.index('is_keyframe') if 'is_keyframe' in header else -1
                start_row = 1
        
        for parts in rows[start_row:]:
            if len(parts) < 4:
                continue
            
            try:
                track_id = int(parts[0])
                frame_idx = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                is_corrected = int(parts[4]) if len(parts) > 4 and parts[4] != '' else 0
                if len(parts) > 7 and has_color:
                    color = (int(parts[5]), int(parts[6]), int(parts[7]))
                else:
                    color = None
                note = parts[note_index].strip() if note_index >= 0 and len(parts) > note_index else ""
                start_frame = int(parts[start_frame_index]) if start_frame_index >= 0 and len(parts) > start_frame_index and parts[start_frame_index] != '' else frame_idx
                is_keyframe = int(parts[keyframe_index]) if keyframe_index >= 0 and len(parts) > keyframe_index and parts[keyframe_index] != '' else 0
            except (ValueError, IndexError):
                continue
            
            if track_id not in imported_tracks:
                imported_tracks[track_id] = Track(
                    track_id=track_id,
                    positions=[],
                    color=color if color else tuple(np.random.randint(0, 255, 3).tolist()),
                    note=note,
                    start_frame=start_frame
                )
            elif color:
                imported_tracks[track_id].color = color
            
            if note:
                imported_tracks[track_id].note = note
            
            track = imported_tracks[track_id]
            track.positions.append((frame_idx, (x, y)))
            track.start_frame = min(track.start_frame, start_frame)
            if is_corrected:
                track.is_manual_corrected[frame_idx] = True
            if is_keyframe:
                track.keyframe_frames[frame_idx] = True
        
        return imported_tracks

    def _export_session_xlsx(self, filepath: str, metadata: dict):
        trajectory_rows = [["track_id", "frame_idx", "x", "y", "is_corrected", "color_r", "color_g", "color_b", "note", "start_frame", "is_keyframe"]]
        for track_id, track in sorted(self.tracks.items()):
            r, g, b = track.color
            for frame_idx, pos in track.positions:
                trajectory_rows.append([
                    track_id, frame_idx, round(pos[0], 2), round(pos[1], 2),
                    int(track.is_manual_corrected.get(frame_idx, False)), r, g, b, track.note, track.start_frame,
                    int(track.keyframe_frames.get(frame_idx, False))
                ])
        
        footprint_rows = [["footprint_id", "frame_idx", "x", "y", "color_r", "color_g", "color_b", "track_id"]]
        for item in metadata.get("footprint_points", []):
            color = item.get("color", [0, 255, 255])
            track_id = item.get("track_id")
            footprint_rows.append([
                item.get("id"),
                item.get("frame_idx"),
                item.get("x"),
                item.get("y"),
                color[0],
                color[1],
                color[2],
                "" if track_id is None else track_id,
            ])
        
        metadata_rows = self._build_metadata_rows(metadata)
        
        sheets = [
            ("Trajectories", trajectory_rows),
            ("Footprints", footprint_rows),
            ("Metadata", metadata_rows),
        ]
        
        workbook_xml = self._build_workbook_xml(sheets)
        workbook_rels = self._build_workbook_rels(len(sheets))
        root_rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '</Relationships>'
        )
        content_types = self._build_content_types(len(sheets))
        styles_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
            '<borders count="1"><border/></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
            '</styleSheet>'
        )
        
        with zipfile.ZipFile(filepath, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('[Content_Types].xml', content_types)
            zf.writestr('_rels/.rels', root_rels)
            zf.writestr('xl/workbook.xml', workbook_xml)
            zf.writestr('xl/_rels/workbook.xml.rels', workbook_rels)
            zf.writestr('xl/styles.xml', styles_xml)
            for idx, (_, rows) in enumerate(sheets, start=1):
                zf.writestr(f'xl/worksheets/sheet{idx}.xml', self._build_sheet_xml(rows))

    def _import_session_xlsx(self, filepath: str) -> Tuple[Dict[int, Track], dict]:
        with zipfile.ZipFile(filepath, 'r') as zf:
            trajectory_rows = self._read_xlsx_sheet_rows(zf, 'xl/worksheets/sheet1.xml')
            sheet_names = set(zf.namelist())
            has_footprints_sheet = 'xl/worksheets/sheet3.xml' in sheet_names
            if has_footprints_sheet:
                footprint_rows = self._read_xlsx_sheet_rows(zf, 'xl/worksheets/sheet2.xml')
                metadata_rows = self._read_xlsx_sheet_rows(zf, 'xl/worksheets/sheet3.xml')
            else:
                footprint_rows = []
                metadata_rows = self._read_xlsx_sheet_rows(zf, 'xl/worksheets/sheet2.xml')
        
        imported_tracks = self._parse_trajectory_rows(trajectory_rows)
        metadata = self._parse_metadata_rows(metadata_rows)
        
        parsed_footprints = self._parse_footprint_rows(footprint_rows)
        if parsed_footprints:
            metadata["footprint_points"] = parsed_footprints
        return imported_tracks, metadata

    def _read_shared_strings(self, zf: zipfile.ZipFile) -> List[str]:
        shared_path = 'xl/sharedStrings.xml'
        if shared_path not in zf.namelist():
            return []

        ns = {'a': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        root = ET.fromstring(zf.read(shared_path))
        values = []
        for item in root.findall('a:si', ns):
            text_parts = []
            for node in item.findall('.//a:t', ns):
                text_parts.append(node.text or '')
            values.append(''.join(text_parts))
        return values

    def _xlsx_column_index(self, ref: str) -> int:
        letters = ''.join(ch for ch in ref if ch.isalpha()).upper()
        result = 0
        for ch in letters:
            result = result * 26 + (ord(ch) - ord('A') + 1)
        return result

    def _read_xlsx_sheet_rows(self, zf: zipfile.ZipFile, sheet_path: str) -> List[List[str]]:
        ns = {'a': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        shared_strings = self._read_shared_strings(zf)
        root = ET.fromstring(zf.read(sheet_path))
        rows = []
        for row in root.findall('.//a:sheetData/a:row', ns):
            current = []
            current_col = 1
            for cell in row.findall('a:c', ns):
                cell_ref = cell.attrib.get('r', '')
                target_col = self._xlsx_column_index(cell_ref) if cell_ref else current_col
                while current_col < target_col:
                    current.append('')
                    current_col += 1

                cell_type = cell.attrib.get('t')
                value = ''
                if cell_type == 'inlineStr':
                    node = cell.find('a:is/a:t', ns)
                    value = node.text if node is not None and node.text is not None else ''
                else:
                    node = cell.find('a:v', ns)
                    raw_value = node.text if node is not None and node.text is not None else ''
                    if cell_type == 's':
                        try:
                            shared_idx = int(raw_value)
                            value = shared_strings[shared_idx] if 0 <= shared_idx < len(shared_strings) else ''
                        except (TypeError, ValueError):
                            value = ''
                    else:
                        value = raw_value

                current.append(value)
                current_col += 1
            rows.append(current)
        return rows

    def _build_sheet_xml(self, rows: List[List[object]]) -> str:
        parts = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
            '<sheetData>',
        ]
        for row_idx, row in enumerate(rows, start=1):
            parts.append(f'<row r="{row_idx}">')
            for col_idx, value in enumerate(row, start=1):
                if value is None:
                    continue
                ref = f'{self._xlsx_column_name(col_idx)}{row_idx}'
                if isinstance(value, bool):
                    value = int(value)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    parts.append(f'<c r="{ref}"><v>{value}</v></c>')
                else:
                    text = escape(str(value))
                    parts.append(f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>')
            parts.append('</row>')
        parts.append('</sheetData></worksheet>')
        return ''.join(parts)

    def _build_workbook_xml(self, sheets: List[Tuple[str, List[List[object]]]]) -> str:
        sheet_xml = []
        for idx, (name, _) in enumerate(sheets, start=1):
            sheet_xml.append(f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>')
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets>' + ''.join(sheet_xml) + '</sheets>'
            '</workbook>'
        )

    def _build_workbook_rels(self, sheet_count: int) -> str:
        rels = []
        for idx in range(1, sheet_count + 1):
            rels.append(
                f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
            )
        rels.append(
            f'<Relationship Id="rId{sheet_count + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + ''.join(rels) +
            '</Relationships>'
        )

    def _build_content_types(self, sheet_count: int) -> str:
        overrides = [
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        ]
        for idx in range(1, sheet_count + 1):
            overrides.append(
                f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            )
        overrides.append('<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>')
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            + ''.join(overrides) +
            '</Types>'
        )

    def _parse_footprint_rows(self, rows: List[List[str]]) -> List[dict]:
        if not rows or not rows[0] or rows[0][0] != 'footprint_id':
            return []
        result = []
        for parts in rows[1:]:
            if len(parts) < 7:
                continue
            try:
                track_id = None if len(parts) < 8 or parts[7] == '' else int(parts[7])
                result.append({
                    "id": int(parts[0]),
                    "frame_idx": int(parts[1]),
                    "x": float(parts[2]),
                    "y": float(parts[3]),
                    "color": [int(parts[4]), int(parts[5]), int(parts[6])],
                    "track_id": track_id,
                })
            except (TypeError, ValueError):
                continue
        return result

    def _build_metadata_rows(self, metadata: dict) -> List[List[object]]:
        rows = [["section", "key", "index", "value1", "value2"]]
        rows.append(["session", "version", "", metadata.get("version", 1), ""])
        
        frame_correction = metadata.get("frame_correction") or {}
        rows.append(["frame_correction", "enabled", "", int(bool(frame_correction.get("enabled", False))), ""])
        for idx, point in enumerate(frame_correction.get("points", []), start=1):
            x = point[0] if len(point) > 0 else ""
            y = point[1] if len(point) > 1 else ""
            rows.append(["frame_correction", "point", idx, x, y])
        
        for idx, x in enumerate(metadata.get("vertical_lines", []), start=1):
            rows.append(["vertical_lines", "x", idx, x, ""])

        display = metadata.get("display") or {}
        rows.append(["display", "trajectory_length_frames", "", int(display.get("trajectory_length_frames", 10)), ""])
        
        return rows

    def _parse_metadata_rows(self, rows: List[List[str]]) -> dict:
        if not rows:
            return {}
        
        if rows[0] and rows[0][0] == "json":
            if len(rows) > 1 and rows[1]:
                try:
                    return json.loads(rows[1][0])
                except Exception:
                    return {}
            return {}
        
        if not rows[0] or rows[0][0] != "section":
            return {}
        
        metadata = {
            "version": 1,
            "frame_correction": {
                "enabled": False,
                "points": [],
            },
            "vertical_lines": [],
            "display": {
                "trajectory_length_frames": 10,
            },
        }
        correction_points = {}
        
        for parts in rows[1:]:
            if len(parts) < 2:
                continue
            section = parts[0]
            key = parts[1]
            index = parts[2] if len(parts) > 2 else ""
            value1 = parts[3] if len(parts) > 3 else ""
            value2 = parts[4] if len(parts) > 4 else ""
            
            try:
                if section == "frame_correction" and key == "enabled":
                    metadata["frame_correction"]["enabled"] = str(value1).strip() in {"1", "true", "True"}
                elif section == "frame_correction" and key == "point":
                    idx = int(index)
                    correction_points[idx] = [float(value1), float(value2)]
                elif section == "vertical_lines" and key == "x":
                    metadata["vertical_lines"].append(float(value1))
                elif section == "session" and key == "version":
                    metadata["version"] = int(value1)
                elif section == "display" and key == "trajectory_length_frames":
                    metadata["display"]["trajectory_length_frames"] = max(1, int(value1))
            except (TypeError, ValueError):
                continue
        
        metadata["frame_correction"]["points"] = [correction_points[idx] for idx in sorted(correction_points.keys())]
        return metadata

    def _xlsx_column_name(self, index: int) -> str:
        result = ""
        while index > 0:
            index, remainder = divmod(index - 1, 26)
            result = chr(65 + remainder) + result
        return result

    def set_track_note(self, track_id: int, note: str):
        if track_id in self.tracks:
            self.tracks[track_id].note = note.strip()

    def set_track_keyframe(self, track_id: int, frame_idx: int, enabled: bool = True) -> bool:
        track = self.tracks.get(track_id)
        if track is None or track.get_position_at(frame_idx) is None:
            return False
        if enabled:
            track.keyframe_frames[frame_idx] = True
        else:
            track.keyframe_frames.pop(frame_idx, None)
        return True

    def has_track_keyframe(self, track_id: int, frame_idx: int) -> bool:
        track = self.tracks.get(track_id)
        if track is None:
            return False
        return bool(track.keyframe_frames.get(frame_idx, False))

    def get_track_keyframe_count(self, track_id: int) -> int:
        track = self.tracks.get(track_id)
        if track is None:
            return 0
        return len(track.keyframe_frames)
    
    def detect_color_blocks(self, frame: np.ndarray, target_color: Tuple[int, int, int], 
                           tolerance: int, min_size: int) -> List[Tuple[int, int, int, int, int, int]]:
        """
        检测帧中符合颜色条件的色块（基于等面积圆直径的过滤）
        
        参数:
            frame: BGR格式的图像
            target_color: 目标BGR颜色 (B, G, R)
            tolerance: 颜色容差
            min_size: 最小像素尺寸（等面积圆的直径），色块面积必须 >= π*(min_size/2)^2
        
        返回:
            色块列表，每个元素为 (x, y, w, h, cx, cy) 即左上角坐标、宽高、中心坐标
        """
        # 转换到HSV颜色空间（对颜色检测更鲁棒）
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 将目标BGR颜色转为HSV
        target_bgr = np.uint8([[target_color]])
        target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)[0][0]
        
        h, s, v = int(target_hsv[0]), int(target_hsv[1]), int(target_hsv[2])
        
        # 计算HSV范围（处理色相的循环）
        h_lower = max(0, h - tolerance)
        h_upper = min(179, h + tolerance)
        s_lower = max(0, s - tolerance)
        s_upper = min(255, s + tolerance)
        v_lower = max(0, v - tolerance)
        v_upper = min(255, v + tolerance)
        
        # 创建掩码
        if h - tolerance < 0 or h + tolerance > 179:
            # 色相跨越边界，需要创建两个范围
            lower1 = np.array([0, s_lower, v_lower])
            upper1 = np.array([h_upper, s_upper, v_upper])
            lower2 = np.array([h_lower, s_lower, v_lower])
            upper2 = np.array([179, s_upper, v_upper])
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            lower = np.array([h_lower, s_lower, v_lower])
            upper = np.array([h_upper, s_upper, v_upper])
            mask = cv2.inRange(hsv, lower, upper)
        
        # 对于小色块，使用更温和的形态学操作
        if min_size <= 10:
            # 小色块：使用3x3核进行轻微的开运算去除单像素噪点
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        else:
            # 大色块：使用5x5核进行标准形态学操作
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 计算面积阈值：等面积圆的面积 = π * (diameter/2)^2
        area_threshold = np.pi * (min_size / 2) ** 2
        
        blocks = []
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 使用面积过滤（等面积圆直径）
            if area < area_threshold:
                continue
            
            # 计算外接矩形（用于返回边界信息）
            x, y, w, h = cv2.boundingRect(contour)
            
            # 使用轮廓矩计算精确的中心点（质心）
            moments = cv2.moments(contour)
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                # Fallback到边界框中心
                cx = x + w // 2
                cy = y + h // 2
            
            blocks.append((x, y, w, h, cx, cy))
        
        return blocks
    
    def match_blocks_to_tracks(self, blocks: List[Tuple], current_frame: int,
                               max_distance: float = 30.0, use_prediction: bool = False) -> Dict[int, int]:
        """
        将色块匹配到现有追踪点（用于自动校正）
        
        参数:
            blocks: 检测到的色块列表
            current_frame: 当前帧索引
            max_distance: 最大匹配距离（默认30像素，校正时应该是微调的）
            use_prediction: 是否使用运动预测（校正时应该为False，只基于当前位置）
        
        返回:
            匹配字典 block_idx -> track_id
        """
        matches = {}
        
        # 获取当前帧活跃的追踪点位置
        track_positions = {}  # track_id -> (x, y)
        for track_id, track in self.tracks.items():
            if track_id in self._active_track_ids:
                # 获取当前位置
                pos = track.get_position_at(current_frame)
                if pos is None and track.positions:
                    _, pos = track.positions[-1]
                
                if pos:
                    if use_prediction:
                        # 只有在明确要求时才使用预测
                        predicted_pos = self._predict_position(track, current_frame)
                        # 限制预测偏移不超过 max_distance/2
                        dx = predicted_pos[0] - pos[0]
                        dy = predicted_pos[1] - pos[1]
                        if abs(dx) > max_distance / 2:
                            dx = max_distance / 2 if dx > 0 else -max_distance / 2
                        if abs(dy) > max_distance / 2:
                            dy = max_distance / 2 if dy > 0 else -max_distance / 2
                        track_positions[track_id] = (pos[0] + dx, pos[1] + dy)
                    else:
                        # 校正时只使用当前位置，不预测
                        track_positions[track_id] = pos
        
        if not track_positions or not blocks:
            return matches
        
        # 简单的贪心匹配：每个追踪点找最近的色块
        track_ids = list(track_positions.keys())
        matched_blocks = set()
        matched_tracks = set()
        
        # 为每个追踪点找到最近的色块
        track_to_best_block = {}
        for track_id in track_ids:
            pos = track_positions[track_id]
            best_block_idx = None
            best_dist = float('inf')
            
            for i, block in enumerate(blocks):
                if i in matched_blocks:
                    continue
                
                bcx, bcy = block[4], block[5]
                dist = np.sqrt((pos[0] - bcx) ** 2 + (pos[1] - bcy) ** 2)
                
                if dist < best_dist and dist <= max_distance:
                    best_dist = dist
                    best_block_idx = i
            
            if best_block_idx is not None:
                track_to_best_block[track_id] = (best_block_idx, best_dist)
        
        # 按距离排序，优先匹配距离近的
        sorted_matches = sorted(track_to_best_block.items(), key=lambda x: x[1][1])
        
        for track_id, (block_idx, dist) in sorted_matches:
            if block_idx in matched_blocks or track_id in matched_tracks:
                continue
            
            matches[block_idx] = track_id
            matched_blocks.add(block_idx)
            matched_tracks.add(track_id)
        
        return matches
    
    def _predict_position(self, track: Track, current_frame: int) -> Tuple[float, float]:
        """
        基于历史位置预测当前帧的位置
        
        返回:
            预测的 (x, y) 位置
        """
        if len(track.positions) < 2:
            # 位置历史不足，返回最后已知位置
            if track.positions:
                return track.positions[-1][1]
            return (0, 0)
        
        # 获取最近两帧的位置计算速度
        (_, pos2), (_, pos1) = track.positions[-1], track.positions[-2]
        
        # 计算速度向量
        vx = pos2[0] - pos1[0]
        vy = pos2[1] - pos1[1]
        
        # 预测当前位置 = 最后位置 + 速度
        predicted_x = pos2[0] + vx
        predicted_y = pos2[1] + vy
        
        return (predicted_x, predicted_y)
    
    def match_blocks_for_correction(self, blocks: List[Tuple], current_frame: int) -> Dict[int, int]:
        """
        将追踪点匹配到包含它的色块（用于自动校正）
        
        逻辑：只有当追踪点当前位置在某个色块内部时，才进行匹配
        
        参数:
            blocks: 检测到的色块列表 (x, y, w, h, cx, cy)
            current_frame: 当前帧索引
        
        返回:
            匹配字典 block_idx -> track_id
        """
        matches = {}
        matched_blocks = set()
        
        # 遍历所有活跃追踪点
        for track_id in self._active_track_ids:
            track = self.tracks.get(track_id)
            if not track:
                continue
            
            # 获取当前位置
            pos = track.get_position_at(current_frame)
            if pos is None and track.positions:
                _, pos = track.positions[-1]
            
            if not pos:
                continue
            
            px, py = pos
            
            # 查找包含该点的色块
            best_block_idx = None
            best_dist_to_center = float('inf')
            
            for i, block in enumerate(blocks):
                if i in matched_blocks:
                    continue
                
                x, y, w, h, cx, cy = block
                
                # 检查点是否在色块边界框内（包含边界）
                if x <= px <= x + w and y <= py <= y + h:
                    # 计算到色块中心的距离
                    dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
                    
                    # 如果当前追踪点在多个色块内，选择最近的中心
                    if dist < best_dist_to_center:
                        best_dist_to_center = dist
                        best_block_idx = i
            
            # 如果找到包含该点的色块，建立匹配
            if best_block_idx is not None:
                matches[best_block_idx] = track_id
                matched_blocks.add(best_block_idx)
        
        return matches
    
    def sample_block_color(self, frame: np.ndarray, pos: Tuple[float, float], 
                          tolerance: int, min_size: int) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int, int, int, int]]]:
        """
        在指定位置采样色块颜色
        
        参数:
            frame: BGR图像
            pos: (x, y) 起始位置
            tolerance: 颜色容差
            min_size: 最小像素尺寸（等面积圆直径）
        
        返回:
            (block_color, block_info) 或 None
            block_color: 色块的平均颜色 (R, G, B)
            block_info: 色块信息 (x, y, w, h, cx, cy)
        """
        h, w = frame.shape[:2]
        x, y = int(pos[0]), int(pos[1])
        
        # 检查位置有效性
        if not (0 <= x < w and 0 <= y < h):
            return None
        
        # 获取起始位置的颜色
        seed_color = frame[y, x]
        seed_bgr = (int(seed_color[0]), int(seed_color[1]), int(seed_color[2]))
        
        # 转换到HSV进行容差判断
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        seed_hsv = cv2.cvtColor(np.uint8([[seed_color]]), cv2.COLOR_BGR2HSV)[0][0]
        
        h_val, s_val, v_val = int(seed_hsv[0]), int(seed_hsv[1]), int(seed_hsv[2])
        
        # 计算HSV范围
        h_lower = max(0, h_val - tolerance)
        h_upper = min(179, h_val + tolerance)
        s_lower = max(0, s_val - tolerance)
        s_upper = min(255, s_val + tolerance)
        v_lower = max(0, v_val - tolerance)
        v_upper = min(255, v_val + tolerance)
        
        # 创建掩码
        lower = np.array([h_lower, s_lower, v_lower])
        upper = np.array([h_upper, s_upper, v_upper])
        mask = cv2.inRange(hsv_frame, lower, upper)
        
        # 使用连通域分析找到包含起始位置的色块
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # 找到包含起始位置的连通域
        target_label = labels[y, x]
        
        if target_label == 0:  # 背景
            return None
        
        # 获取该连通域的统计信息
        x1, y1, bw, bh, area = stats[target_label]
        cx, cy = int(centroids[target_label][0]), int(centroids[target_label][1])
        
        # 检查面积是否满足要求（等面积圆直径）
        area_threshold = np.pi * (min_size / 2) ** 2
        if area < area_threshold:
            return None
        
        # 计算色块的平均颜色
        mask_single = (labels == target_label).astype(np.uint8) * 255
        mean_color = cv2.mean(frame, mask=mask_single)
        block_color = (int(mean_color[2]), int(mean_color[1]), int(mean_color[0]))  # BGR to RGB
        
        block_info = (x1, y1, bw, bh, cx, cy)
        
        return block_color, block_info
    
    def find_block_near_position(self, frame: np.ndarray, pos: Tuple[float, float],
                                 target_color: Tuple[int, int, int],
                                 tolerance: int, min_size: int,
                                 search_radius: int = 50) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int, int, int, int]]]:
        """
        在指定位置附近使用统一基准颜色搜索色块（局部搜索）
        
        参数:
            frame: BGR图像
            pos: (x, y) 搜索中心位置
            target_color: 基准颜色 (B, G, R)
            tolerance: 颜色容差
            min_size: 最小像素尺寸
            search_radius: 搜索半径（像素）
        
        返回:
            (block_color, block_info) 或 None
        """
        h, w = frame.shape[:2]
        px, py = int(pos[0]), int(pos[1])
        
        # 计算搜索区域（以pos为中心的正方形）
        x1 = max(0, px - search_radius)
        y1 = max(0, py - search_radius)
        x2 = min(w, px + search_radius)
        y2 = min(h, py + search_radius)
        
        if x1 >= x2 or y1 >= y2:
            return None
        
        # 裁剪搜索区域
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        
        # 在ROI中使用基准颜色检测色块
        roi_blocks = self.detect_color_blocks(roi, target_color, tolerance, min_size)
        
        if not roi_blocks:
            return None
        
        # 找到离搜索中心最近的色块
        best_block = None
        best_dist = float('inf')
        
        for block in roi_blocks:
            # 色块中心坐标需要加上ROI的偏移
            cx = block[4] + x1
            cy = block[5] + y1
            
            dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_block = (block[0] + x1, block[1] + y1, block[2], block[3], cx, cy)
        
        if best_block is None:
            return None
        
        # 计算色块的真实颜色（用于显示）
        bx, by, bw, bh, bcx, bcy = best_block
        # 确保不越界
        bx = max(0, bx)
        by = max(0, by)
        bw = min(w - bx, bw)
        bh = min(h - by, bh)
        
        if bw <= 0 or bh <= 0:
            return None
        
        block_roi = frame[by:by+bh, bx:bx+bw]
        mean_color = cv2.mean(block_roi)
        block_color = (int(mean_color[2]), int(mean_color[1]), int(mean_color[0]))  # BGR to RGB
        
        return block_color, best_block
    
    def process_color_tracking(self, frame: np.ndarray, frame_idx: int,
                               target_color: Tuple[int, int, int],
                               tolerance: int, min_size: int,
                               auto_add: bool, auto_correct: bool, auto_end: bool) -> dict:
        """
        处理色块自动追踪（校正时使用统一基准颜色局部搜索）
        
        返回:
            {
                'added': [(track_id, cx, cy), ...],  # 新增的追踪点
                'corrected': [(track_id, cx, cy), ...],  # 校正的追踪点
                'ended': [track_id, ...],  # 结束的追踪点
                'blocks': [...],  # 检测到的色块（仅用于新增）
                'on_block_status': {track_id: (bool, color), ...}  # 追踪点是否在色块上，如果在则包含色块颜色
            }
        """
        result = {
            'added': [],
            'corrected': [],
            'ended': [],
            'blocks': [],
            'on_block_status': {}
        }
        
        # === 第一步：使用统一基准颜色全局检测色块 ===
        all_blocks = self.detect_color_blocks(frame, target_color, tolerance, min_size)
        result['blocks'] = all_blocks
        
        # === 第二步：自动添加新追踪点 ===
        if auto_add:
            for block in all_blocks:
                x, y, w, h, cx, cy = block
                # 检查该位置是否已有追踪点
                too_close = False
                for track_id in self._active_track_ids:
                    track = self.tracks.get(track_id)
                    if track and track.positions:
                        last_pos = track.positions[-1][1]
                        dist = np.sqrt((last_pos[0] - cx) ** 2 + (last_pos[1] - cy) ** 2)
                        if dist < min_size:  # 如果已有追踪点在附近，不添加
                            too_close = True
                            break
                
                if not too_close:
                    # 添加新追踪点
                    track_id = self.add_track_point(frame, frame_idx, (cx, cy), color=target_color)
                    
                    # 计算该色块的真实颜色（用于显示）
                    bx, by, bw, bh = x, y, w, h
                    h_frame, w_frame = frame.shape[:2]
                    bx = max(0, bx)
                    by = max(0, by)
                    bw = min(w_frame - bx, bw)
                    bh = min(h_frame - by, bh)
                    if bw > 0 and bh > 0:
                        block_roi = frame[by:by+bh, bx:bx+bw]
                        mean_color = cv2.mean(block_roi)
                        self.tracks[track_id].block_color = (int(mean_color[2]), int(mean_color[1]), int(mean_color[0]))
                    
                    result['added'].append((track_id, cx, cy))
                    result['on_block_status'][track_id] = (True, self.tracks[track_id].block_color)
        
        # === 第二步：自动校正和结束 ===
        # 只有追踪点在色块内部时，才校正到色块中心
        if auto_correct or auto_end:
            for track_id in list(self._active_track_ids):
                track = self.tracks.get(track_id)
                if not track:
                    continue
                
                # 获取当前位置
                pos = track.get_position_at(frame_idx)
                if pos is None and track.positions:
                    _, pos = track.positions[-1]
                
                if not pos:
                    continue
                
                px, py = pos
                
                # 查找包含该追踪点的色块（点在色块边界框内）
                containing_block = None
                containing_block_color = None
                
                for block in all_blocks:
                    bx, by, bw, bh, bcx, bcy = block
                    # 检查点是否在色块边界框内
                    if bx <= px <= bx + bw and by <= py <= by + bh:
                        # 计算该色块的颜色（用于显示）
                        h_frame, w_frame = frame.shape[:2]
                        bx_safe = max(0, bx)
                        by_safe = max(0, by)
                        bw_safe = min(w_frame - bx_safe, bw)
                        bh_safe = min(h_frame - by_safe, bh)
                        if bw_safe > 0 and bh_safe > 0:
                            block_roi = frame[by_safe:by_safe+bh_safe, bx_safe:bx_safe+bw_safe]
                            mean_color = cv2.mean(block_roi)
                            block_color = (int(mean_color[2]), int(mean_color[1]), int(mean_color[0]))
                            
                            containing_block = block
                            containing_block_color = block_color
                            break  # 找到一个包含该点的色块即可
                
                if containing_block:
                    # 追踪点在色块内部，校正到色块中心
                    cx, cy = containing_block[4], containing_block[5]
                    
                    # 记录色块颜色（用于显示）
                    if track.block_color is None:
                        track.block_color = containing_block_color
                    
                    if auto_correct:
                        self.correct_track_position(track_id, frame_idx, (cx, cy))
                        result['corrected'].append((track_id, cx, cy))
                    
                    result['on_block_status'][track_id] = (True, containing_block_color)
                else:
                    # 追踪点不在任何色块内部
                    result['on_block_status'][track_id] = (False, None)
                    if auto_end:
                        self.end_track_point(track_id, frame_idx, pos)
                        result['ended'].append(track_id)
        
        return result
