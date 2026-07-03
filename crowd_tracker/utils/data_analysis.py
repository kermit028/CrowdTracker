import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from core.tracker import Track
from utils.simple_xlsx import write_simple_xlsx


SCENE_SINGLE = "single"
SCENE_QUEUE = "queue"
SCENE_CROWD = "crowd"


class AnalysisError(Exception):
    pass


@dataclass
class TrackAnalysis:
    track_id: int
    note: str
    points: List[dict]
    footprints: List[dict]
    speed_samples: List[float]
    stride_samples: List[float]
    interval_samples: List[float]
    mean_speed: Optional[float]
    speed_variance: Optional[float]
    min_speed: Optional[float]
    max_speed: Optional[float]
    mean_stride: Optional[float]
    mean_interval: Optional[float]
    direction_dx_m: float
    peak_valley_samples: List[float]
    mean_peak_valley: Optional[float]
    peak_height_samples: List[dict]
    mean_peak_height: Optional[float]
    footprint_baseline: Optional[Tuple[float, float]]


def export_analysis_xlsx(
    filepath: str,
    tracks: Dict[int, Track],
    footprints: Sequence[dict],
    vertical_lines: Sequence[float],
    fps: float,
    scene_mode: str,
    video_name: str = "",
) -> None:
    workbook = build_analysis_workbook(
        tracks=tracks,
        footprints=footprints,
        vertical_lines=vertical_lines,
        fps=fps,
        scene_mode=scene_mode,
        video_name=video_name,
    )
    write_simple_xlsx(filepath, workbook)


def analyze_tracks_for_scene(
    tracks: Dict[int, Track],
    footprints: Sequence[dict],
    vertical_lines: Sequence[float],
    fps: float,
) -> Tuple[Tuple[float, float, float], List[TrackAnalysis]]:
    if fps <= 0:
        raise AnalysisError("FPS 无效，无法计算速度和时间间隔。")

    region_start_px, region_end_px, meter_per_pixel = _resolve_region(vertical_lines)
    track_analyses = _analyze_tracks(
        tracks,
        footprints,
        fps,
        region_start_px,
        region_end_px,
        meter_per_pixel,
    )
    return (region_start_px, region_end_px, meter_per_pixel), track_analyses


def build_analysis_workbook(
    tracks: Dict[int, Track],
    footprints: Sequence[dict],
    vertical_lines: Sequence[float],
    fps: float,
    scene_mode: str,
    video_name: str = "",
) -> List[Tuple[str, List[List[object]]]]:
    if scene_mode == SCENE_CROWD:
        raise AnalysisError("人群场景暂不生成处理数据。")
    if fps <= 0:
        raise AnalysisError("FPS 无效，无法计算速度和时间间隔。")

    (region_start_px, region_end_px, meter_per_pixel), track_analyses = analyze_tracks_for_scene(
        tracks=tracks,
        footprints=footprints,
        vertical_lines=vertical_lines,
        fps=fps,
    )

    sheets: List[Tuple[str, List[List[object]]]] = [
        (
            "Overview",
            _build_overview_rows(
                video_name=video_name,
                fps=fps,
                scene_mode=scene_mode,
                region_start_px=region_start_px,
                region_end_px=region_end_px,
                meter_per_pixel=meter_per_pixel,
                track_analyses=track_analyses,
            ),
        ),
        ("TrackSummary", _build_track_summary_rows(track_analyses)),
        ("TrackTrajectory", _build_track_trajectory_rows(track_analyses)),
        ("Footprints", _build_footprint_rows(track_analyses)),
    ]

    if scene_mode == SCENE_QUEUE:
        sheets.append(("QueueSpacing", _build_queue_rows(track_analyses)))

    return sheets


def _resolve_region(vertical_lines: Sequence[float]) -> Tuple[float, float, float]:
    lines = sorted(float(x) for x in vertical_lines)
    if len(lines) < 4:
        raise AnalysisError("需要先标记至少 4 条竖线，才能识别中间 5m 分析区间。")

    region_start_px = lines[1]
    region_end_px = lines[2]
    width_px = abs(region_end_px - region_start_px)
    if width_px <= 1e-6:
        raise AnalysisError("中间两条竖线重合，无法建立 5m 区间。")

    return region_start_px, region_end_px, 5.0 / width_px


def _analyze_tracks(
    tracks: Dict[int, Track],
    footprints: Sequence[dict],
    fps: float,
    region_start_px: float,
    region_end_px: float,
    meter_per_pixel: float,
) -> List[TrackAnalysis]:
    min_x = min(region_start_px, region_end_px)
    max_x = max(region_start_px, region_end_px)
    analysis_by_track: List[TrackAnalysis] = []

    footprints_by_track: Dict[int, List[dict]] = {}
    for item in footprints:
        track_id = item.get("track_id")
        if track_id is None:
            continue
        try:
            normalized_track_id = int(track_id)
        except (TypeError, ValueError):
            continue
        footprints_by_track.setdefault(normalized_track_id, []).append(item)

    for track_id, track in sorted(tracks.items()):
        filtered_positions = []
        for frame_idx, (x_px, y_px) in sorted(track.positions, key=lambda item: item[0]):
            if min_x <= x_px <= max_x:
                filtered_positions.append({
                    "frame_idx": int(frame_idx),
                    "time_s": round(frame_idx / fps, 4),
                    "x_m": round((x_px - region_start_px) * meter_per_pixel, 4),
                    "y_m": round(-y_px * meter_per_pixel, 4),
                    "is_corrected": int(bool(track.is_manual_corrected.get(frame_idx, False))),
                    "is_keyframe": int(bool(track.keyframe_frames.get(frame_idx, False))),
                    "x_px": float(x_px),
                    "y_px": float(y_px),
                })

        speed_samples = _build_speed_samples(filtered_positions, fps)
        speed_stats = _build_basic_stats(speed_samples)

        peak_valley_samples = _detect_peak_valley(filtered_positions, fps)
        mean_peak_valley = _safe_mean(peak_valley_samples)

        related_footprints = []
        for item in sorted(footprints_by_track.get(track_id, []), key=lambda row: (row.get("frame_idx", 0), row.get("id", 0))):
            x_px = float(item.get("x", 0.0))
            y_px = float(item.get("y", 0.0))
            if not (min_x <= x_px <= max_x):
                continue
            related_footprints.append({
                "footprint_id": int(item.get("id", 0)),
                "frame_idx": int(item.get("frame_idx", 0)),
                "time_s": round(int(item.get("frame_idx", 0)) / fps, 4),
                "x_m": round((x_px - region_start_px) * meter_per_pixel, 4),
                "y_m": round(-y_px * meter_per_pixel, 4),
                "x_px": x_px,
                "y_px": y_px,
                "color": tuple(item.get("color", (0, 255, 255))),
            })

        stride_samples = _build_stride_samples(related_footprints)
        interval_samples = _build_interval_samples(related_footprints)
        direction_dx_m = 0.0
        if len(filtered_positions) >= 2:
            direction_dx_m = filtered_positions[-1]["x_m"] - filtered_positions[0]["x_m"]

        footprint_baseline = _fit_footprint_baseline(related_footprints)
        peak_indices = _find_peak_indices(filtered_positions, fps)
        peak_height_samples = _compute_peak_heights(filtered_positions, peak_indices, footprint_baseline)
        mean_peak_height = _safe_mean([sample["height_m"] for sample in peak_height_samples])

        analysis_by_track.append(
            TrackAnalysis(
                track_id=track_id,
                note=_normalize_note(track.note),
                points=filtered_positions,
                footprints=related_footprints,
                speed_samples=speed_samples,
                stride_samples=stride_samples,
                interval_samples=interval_samples,
                mean_speed=speed_stats["mean"],
                speed_variance=speed_stats["variance"],
                min_speed=speed_stats["min"],
                max_speed=speed_stats["max"],
                mean_stride=_safe_mean(stride_samples),
                mean_interval=_safe_mean(interval_samples),
                direction_dx_m=direction_dx_m,
                peak_valley_samples=peak_valley_samples,
                mean_peak_valley=mean_peak_valley,
                peak_height_samples=peak_height_samples,
                mean_peak_height=mean_peak_height,
                footprint_baseline=footprint_baseline,
            )
        )

    return analysis_by_track


def _build_speed_samples(
    points: Sequence[dict],
    fps: float,
    window_s: float = 0.2,
) -> List[float]:
    """基于固定时间窗口计算速度样本，降低单帧像素离散带来的噪声。"""
    if len(points) < 2 or fps <= 0 or window_s <= 0:
        return []

    start_time = points[0]["time_s"]
    end_time = points[-1]["time_s"]
    if end_time <= start_time:
        return []

    samples = []
    current_time = start_time
    while current_time + window_s <= end_time:
        next_time = current_time + window_s
        p_start = _interpolate_point_at_time(points, current_time)
        p_end = _interpolate_point_at_time(points, next_time)
        if p_start is None or p_end is None:
            current_time += window_s
            continue
        dx = p_end["x_m"] - p_start["x_m"]
        distance_m = abs(dx)
        speed_m_s = distance_m / window_s
        samples.append(round(speed_m_s, 6))
        current_time += window_s
    return samples


def _interpolate_point_at_time(points: Sequence[dict], target_time: float) -> Optional[dict]:
    """按时间 target_time 在轨迹点序列中线性插值得到位置。"""
    if not points:
        return None
    if target_time <= points[0]["time_s"]:
        return points[0]
    if target_time >= points[-1]["time_s"]:
        return points[-1]

    for prev, current in zip(points, points[1:]):
        if prev["time_s"] <= target_time <= current["time_s"]:
            dt = current["time_s"] - prev["time_s"]
            if dt <= 0:
                return prev
            ratio = (target_time - prev["time_s"]) / dt
            return {
                "x_m": prev["x_m"] + ratio * (current["x_m"] - prev["x_m"]),
                "y_m": prev["y_m"] + ratio * (current["y_m"] - prev["y_m"]),
            }
    return points[-1]


def _build_stride_samples(footprints: Sequence[dict]) -> List[float]:
    samples = []
    for prev, current in zip(footprints, footprints[1:]):
        dx = current["x_m"] - prev["x_m"]
        dy = current["y_m"] - prev["y_m"]
        samples.append(round(math.hypot(dx, dy), 6))
    return samples


def _build_interval_samples(footprints: Sequence[dict]) -> List[float]:
    samples = []
    for prev, current in zip(footprints, footprints[1:]):
        dt = current["time_s"] - prev["time_s"]
        if dt > 0:
            samples.append(round(dt, 6))
    return samples


def _find_peak_indices(
    points: Sequence[dict],
    fps: float,
    min_peak_distance_s: float = 0.15,
) -> List[int]:
    """检测 y_m 序列中的局部峰值索引（头部高点）。"""
    if len(points) < 3 or fps <= 0:
        return []

    n = len(points)
    y_values = [p["y_m"] for p in points]
    min_samples = max(2, int(min_peak_distance_s * fps / 2))

    peak_indices = []
    for i in range(min_samples, n - min_samples):
        current_y = y_values[i]
        left_max = max(y_values[i - min_samples:i])
        right_max = max(y_values[i + 1:i + 1 + min_samples])
        if current_y > left_max and current_y >= right_max:
            peak_indices.append(i)

    # 按最小距离去重（保留较高的峰）
    filtered = []
    for idx in sorted(peak_indices, key=lambda i: y_values[i], reverse=True):
        if all(abs(idx - kept) > min_samples for kept in filtered):
            filtered.append(idx)
    return sorted(filtered)


def _find_valley_indices(
    points: Sequence[dict],
    peak_indices: Sequence[int],
) -> List[int]:
    """在每对相邻峰值之间的最低谷位置。"""
    if len(peak_indices) < 2:
        return []

    y_values = [p["y_m"] for p in points]
    valley_indices = []
    for prev_peak, next_peak in zip(peak_indices, peak_indices[1:]):
        segment = y_values[prev_peak:next_peak + 1]
        local_min_offset = segment.index(min(segment))
        valley_indices.append(prev_peak + local_min_offset)
    return valley_indices


def _detect_peak_valley(
    points: Sequence[dict],
    fps: float,
    min_peak_distance_s: float = 0.15,
) -> List[float]:
    """计算相邻峰-谷之间的竖向距离（峰谷差）样本。"""
    peak_indices = _find_peak_indices(points, fps, min_peak_distance_s)
    valley_indices = _find_valley_indices(points, peak_indices)
    if not peak_indices or not valley_indices:
        return []

    y_values = [p["y_m"] for p in points]
    amplitudes = []
    # 每个峰值与左右最近的谷值取平均差
    for i, peak_idx in enumerate(peak_indices):
        peak_y = y_values[peak_idx]
        left_valley_y = None
        right_valley_y = None
        if i > 0:
            left_valley_y = y_values[valley_indices[i - 1]]
        if i < len(valley_indices):
            right_valley_y = y_values[valley_indices[i]]

        diffs = []
        if left_valley_y is not None:
            diffs.append(abs(peak_y - left_valley_y))
        if right_valley_y is not None:
            diffs.append(abs(peak_y - right_valley_y))
        if diffs:
            amplitudes.append(round(sum(diffs) / len(diffs), 6))

    return amplitudes


def _fit_footprint_baseline(
    footprints: Sequence[dict],
) -> Optional[Tuple[float, float]]:
    """用脚印点线性拟合行走基准线 y = a * x + b，返回 (a, b)。"""
    if len(footprints) < 2:
        return None

    xs = [fp["x_m"] for fp in footprints]
    ys = [fp["y_m"] for fp in footprints]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if abs(denominator) < 1e-12:
        return None
    a = numerator / denominator
    b = mean_y - a * mean_x
    return round(a, 8), round(b, 8)


def _compute_peak_heights(
    points: Sequence[dict],
    peak_indices: Sequence[int],
    baseline: Optional[Tuple[float, float]],
) -> List[dict]:
    """计算每个峰值相对脚印基准线的高度差。"""
    if baseline is None or not peak_indices:
        return []

    a, b = baseline
    samples = []
    for idx in peak_indices:
        point = points[idx]
        x_m = point["x_m"]
        y_m = point["y_m"]
        baseline_y = a * x_m + b
        height_m = round(y_m - baseline_y, 6)
        samples.append({
            "frame_idx": point["frame_idx"],
            "time_s": point["time_s"],
            "x_m": x_m,
            "y_m": y_m,
            "baseline_y_m": round(baseline_y, 6),
            "height_m": height_m,
        })
    return samples


def _build_basic_stats(values: Sequence[float]) -> dict:
    if not values:
        return {"mean": None, "variance": None, "min": None, "max": None}
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return {
        "mean": round(mean_value, 6),
        "variance": round(variance, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _normalize_note(note: str) -> str:
    normalized = (note or "").strip()
    return normalized if normalized else "null"


def _build_overview_rows(
    video_name: str,
    fps: float,
    scene_mode: str,
    region_start_px: float,
    region_end_px: float,
    meter_per_pixel: float,
    track_analyses: Sequence[TrackAnalysis],
) -> List[List[object]]:
    rows: List[List[object]] = [
        ["item", "value"],
        ["video_name", video_name or ""],
        ["scene_mode", _scene_mode_label(scene_mode)],
        ["fps", round(fps, 6)],
        ["region_left_px", round(min(region_start_px, region_end_px), 4)],
        ["region_right_px", round(max(region_start_px, region_end_px), 4)],
        ["region_width_m", 5.0],
        ["meter_per_pixel", round(meter_per_pixel, 8)],
        ["track_count_total", len(track_analyses)],
        ["track_count_with_points", sum(1 for item in track_analyses if item.points)],
        ["track_count_with_footprints", sum(1 for item in track_analyses if item.footprints)],
    ]
    return rows


def _build_track_summary_rows(track_analyses: Sequence[TrackAnalysis]) -> List[List[object]]:
    rows: List[List[object]] = [[
        "track_id", "note", "point_count_5m", "duration_s", "mean_speed_m_s",
        "speed_variance", "speed_min_m_s", "speed_max_m_s",
        "footprint_count_5m", "mean_stride_m", "mean_interval_s",
        "mean_peak_valley_m", "mean_peak_height_m", "status"
    ]]

    for item in track_analyses:
        duration_s = None
        if len(item.points) >= 2:
            duration_s = round(item.points[-1]["time_s"] - item.points[0]["time_s"], 6)
        status = "ok" if item.points else "no_points_in_5m"
        rows.append([
            item.track_id,
            item.note,
            len(item.points),
            duration_s,
            item.mean_speed,
            item.speed_variance,
            item.min_speed,
            item.max_speed,
            len(item.footprints),
            item.mean_stride,
            item.mean_interval,
            item.mean_peak_valley,
            item.mean_peak_height,
            status,
        ])
    return rows


def _build_track_trajectory_rows(track_analyses: Sequence[TrackAnalysis]) -> List[List[object]]:
    rows: List[List[object]] = []
    for item in track_analyses:
        rows.append(["track_id", item.track_id, "note", item.note])
        rows.append([
            "summary", "point_count_5m", "mean_speed_m_s", "speed_variance",
            "speed_min_m_s", "speed_max_m_s"
        ])
        rows.append([
            "",
            len(item.points),
            item.mean_speed,
            item.speed_variance,
            item.min_speed,
            item.max_speed,
        ])
        rows.append([
            "frame_idx", "time_s", "x_m", "y_m", "is_corrected", "is_keyframe"
        ])
        if item.points:
            for point in item.points:
                rows.append([
                    point["frame_idx"],
                    point["time_s"],
                    point["x_m"],
                    point["y_m"],
                    point["is_corrected"],
                    point["is_keyframe"],
                ])
        else:
            rows.append(["no data in 5m zone", "", "", "", "", ""])
        rows.append([])
    return rows


def _build_footprint_rows(track_analyses: Sequence[TrackAnalysis]) -> List[List[object]]:
    rows: List[List[object]] = []
    for item in track_analyses:
        rows.append(["track_id", item.track_id, "note", item.note])
        rows.append([
            "summary", "footprint_count_5m", "mean_stride_m", "mean_interval_s",
            "mean_peak_valley_m", "mean_peak_height_m"
        ])
        rows.append([
            "",
            len(item.footprints),
            item.mean_stride,
            item.mean_interval,
            item.mean_peak_valley,
            item.mean_peak_height,
        ])
        rows.append(["footprint_id", "frame_idx", "time_s", "x_m", "y_m"])
        if item.footprints:
            for footprint in item.footprints:
                rows.append([
                    footprint["footprint_id"],
                    footprint["frame_idx"],
                    footprint["time_s"],
                    footprint["x_m"],
                    footprint["y_m"],
                ])
            rows.append([])
            rows.append(["stride_index", "stride_m", "interval_s"])
            max_len = max(len(item.stride_samples), len(item.interval_samples))
            for idx in range(max_len):
                rows.append([
                    idx + 1,
                    item.stride_samples[idx] if idx < len(item.stride_samples) else None,
                    item.interval_samples[idx] if idx < len(item.interval_samples) else None,
                ])
            if item.peak_height_samples:
                rows.append([])
                rows.append([
                    "peak_index", "frame_idx", "time_s", "x_m", "y_m",
                    "baseline_y_m", "height_above_baseline_m"
                ])
                for idx, peak in enumerate(item.peak_height_samples, start=1):
                    rows.append([
                        idx,
                        peak["frame_idx"],
                        peak["time_s"],
                        peak["x_m"],
                        peak["y_m"],
                        peak["baseline_y_m"],
                        peak["height_m"],
                    ])
        else:
            rows.append(["no footprints in 5m zone", "", "", "", ""])
        rows.append([])
    return rows


def _build_queue_rows(track_analyses: Sequence[TrackAnalysis]) -> List[List[object]]:
    direction_sign = _resolve_queue_direction(track_analyses)
    detail_rows = _build_queue_detail_rows(track_analyses, direction_sign)

    rows: List[List[object]] = [[
        "track_id", "note", "front_track_id", "sample_count",
        "mean_gap_m", "gap_variance", "gap_min_m", "gap_max_m"
    ]]

    grouped: Dict[int, List[dict]] = {}
    for row in detail_rows:
        grouped.setdefault(row["track_id"], []).append(row)

    note_map = {item.track_id: item.note for item in track_analyses}
    for track_id in sorted(note_map.keys()):
        track_rows = grouped.get(track_id, [])
        gaps = [row["gap_m"] for row in track_rows]
        stats = _build_basic_stats(gaps)
        front_ids = sorted({row["front_track_id"] for row in track_rows})
        rows.append([
            track_id,
            note_map.get(track_id, "null"),
            ",".join(str(front_id) for front_id in front_ids) if front_ids else "",
            len(track_rows),
            stats["mean"],
            stats["variance"],
            stats["min"],
            stats["max"],
        ])

    rows.append([])
    rows.append(["frame_idx", "time_s", "track_id", "front_track_id", "track_x_m", "front_x_m", "gap_m"])
    if detail_rows:
        for row in detail_rows:
            rows.append([
                row["frame_idx"],
                row["time_s"],
                row["track_id"],
                row["front_track_id"],
                row["track_x_m"],
                row["front_x_m"],
                row["gap_m"],
            ])
    else:
        rows.append(["no queue relation samples", "", "", "", "", "", ""])
    return rows


def _resolve_queue_direction(track_analyses: Sequence[TrackAnalysis]) -> int:
    total_dx = sum(item.direction_dx_m for item in track_analyses if len(item.points) >= 2)
    return -1 if total_dx < 0 else 1


def _build_queue_detail_rows(track_analyses: Sequence[TrackAnalysis], direction_sign: int) -> List[dict]:
    frames: Dict[int, List[dict]] = {}
    for item in track_analyses:
        for point in item.points:
            frames.setdefault(point["frame_idx"], []).append({
                "track_id": item.track_id,
                "time_s": point["time_s"],
                "x_m": point["x_m"],
            })

    detail_rows: List[dict] = []
    for frame_idx in sorted(frames.keys()):
        frame_items = frames[frame_idx]
        if len(frame_items) < 2:
            continue
        ordered = sorted(frame_items, key=lambda row: row["x_m"], reverse=(direction_sign < 0))
        for idx, current in enumerate(ordered[:-1]):
            front = ordered[idx + 1]
            gap_m = round(abs(front["x_m"] - current["x_m"]), 6)
            detail_rows.append({
                "frame_idx": frame_idx,
                "time_s": current["time_s"],
                "track_id": current["track_id"],
                "front_track_id": front["track_id"],
                "track_x_m": current["x_m"],
                "front_x_m": front["x_m"],
                "gap_m": gap_m,
            })
    return detail_rows


def _scene_mode_label(scene_mode: str) -> str:
    mapping = {
        SCENE_SINGLE: "单人场景",
        SCENE_QUEUE: "队列场景",
        SCENE_CROWD: "人群场景",
    }
    return mapping.get(scene_mode, scene_mode)
