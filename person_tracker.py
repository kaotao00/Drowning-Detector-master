"""简单的多人位置关联与风险提示；不替代经过验证的救生系统。"""
from collections import deque
from dataclasses import dataclass, field
from math import hypot


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def size(box):
    return max(box[2] - box[0], 1), max(box[3] - box[1], 1)


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    aw, ah = size(a)
    bw, bh = size(b)
    return intersection / max(aw * ah + bw * bh - intersection, 1)


def motion_features(history, window, movement_threshold):
    """计算滑动时间窗口内的特征；score 是启发式评分，不是概率。"""
    if len(history) < 2 or history[-1][0] - history[0][0] < window:
        return None
    cutoff = history[-1][0] - window
    samples = list(history)
    # 将起点插值到窗口边界，避免窗口外的旧位移影响结果。
    if samples[0][0] < cutoff:
        a, b = samples[0], samples[1]
        ratio = (cutoff - a[0]) / (b[0] - a[0])
        samples[0] = (cutoff,) + tuple(a[i] + ratio * (b[i] - a[i]) for i in range(1, 4))
    scale = sum(sample[3] for sample in samples) / len(samples)
    xs = [sample[1] for sample in samples]
    ys = [sample[2] for sample in samples]
    extent = hypot(max(xs) - min(xs), max(ys) - min(ys)) / max(scale, 1)
    total_distance = stationary_time = total_time = 0.0
    speed_threshold = movement_threshold / window
    for a, b in zip(samples, samples[1:]):
        dt = b[0] - a[0]
        if dt <= 0:
            continue
        distance = hypot(b[1] - a[1], b[2] - a[2]) / max((a[3] + b[3]) / 2, 1)
        total_distance += distance
        total_time += dt
        if distance / dt <= speed_threshold:
            stationary_time += dt
    if total_time <= 0:
        return None
    mean_speed = total_distance / total_time
    stationary_fraction = stationary_time / total_time
    # 分数仅描述低运动程度，权重和阈值未经训练或概率校准。
    low_speed = max(0.0, 1.0 - mean_speed / max(speed_threshold, 1e-9))
    low_extent = max(0.0, 1.0 - extent / max(movement_threshold, 1e-9))
    score = 0.5 * stationary_fraction + 0.3 * low_speed + 0.2 * low_extent
    return {'mean_speed': mean_speed, 'extent': extent,
            'stationary_fraction': stationary_fraction, 'score': score}


@dataclass
class Track:
    id: int
    box: list
    confidence: float
    last_seen: float
    history: deque = field(default_factory=deque)
    visible: bool = True
    risk: bool = False
    motion: float = 0.0
    features: dict | None = None


class PersonTracker:
    def __init__(self, max_missing=2.0, max_observation_gap=1.5,
                 window=10.0, movement_threshold=0.30, vertical_ratio=1.2,
                 waterline=None, enable_risk=False, score_threshold=0.75):
        self.max_missing = max_missing
        self.max_observation_gap = max_observation_gap
        self.window = window
        self.movement_threshold = movement_threshold
        self.vertical_ratio = vertical_ratio
        self.waterline = waterline
        self.enable_risk = enable_risk
        self.score_threshold = score_threshold
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, now, frame_height):
        # 先过期再关联，避免长时间消失后复用旧 ID。
        self.tracks = {k: t for k, t in self.tracks.items()
                       if now - t.last_seen <= self.max_missing}
        for track in self.tracks.values():
            track.visible = False
            track.risk = False
            track.features = None

        candidates = []
        for track in self.tracks.values():
            for index, (box, confidence) in enumerate(detections):
                tw, th = size(track.box)
                bw, bh = size(box)
                a, b = center(track.box), center(box)
                distance = hypot(a[0] - b[0], a[1] - b[1]) / max(hypot(tw, th), hypot(bw, bh), 1)
                overlap = iou(track.box, box)
                if overlap >= 0.1 or distance <= 0.75:
                    candidates.append((distance + (1 - overlap), track.id, index))

        used_tracks, used_detections = set(), set()
        for _, track_id, index in sorted(candidates):
            if track_id in used_tracks or index in used_detections:
                continue
            track = self.tracks[track_id]
            box, confidence = detections[index]
            # 观察间隔过长时重新累计，避免对缺失时段作推断。
            if now - track.last_seen > self.max_observation_gap:
                track.history.clear()
            track.box, track.confidence = box, confidence
            track.last_seen, track.visible = now, True
            used_tracks.add(track_id)
            used_detections.add(index)

        for index, (box, confidence) in enumerate(detections):
            if index not in used_detections:
                track = Track(self.next_id, box, confidence, now)
                self.tracks[track.id] = track
                self.next_id += 1

        for track in self.tracks.values():
            if not track.visible:
                continue
            x, y = center(track.box)
            w, h = size(track.box)
            track.history.append((now, x, y, hypot(w, h)))
            # 保留窗口开始前最近一次观察，以覆盖完整窗口。
            while len(track.history) > 1 and track.history[1][0] <= now - self.window:
                track.history.popleft()
            track.features = motion_features(track.history, self.window, self.movement_threshold)
            track.motion = track.features['extent'] if track.features else 0.0
            vertical = h / w >= self.vertical_ratio
            # 水面线只是画面中的参考线，不能证明人在水下。
            below_line = self.waterline is not None and y >= self.waterline * frame_height
            track.risk = bool(self.enable_risk and below_line and vertical and track.features
                              and track.features['score'] >= self.score_threshold)
        return list(self.tracks.values())
