import math
import time
import ctypes
from collections import deque
from ctypes import wintypes
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import mss
from ultralytics import YOLO

# -------------------- Win32 helpers --------------------
user32 = ctypes.windll.user32
GA_ROOT = 2
VK_LBUTTON = 0x01

# -------------------- Tunables --------------------
CONF_THRESHOLD = 0.55
PERSON_CLASS_ID = 0

# speed
IMGSZ = 640           # YOLO input size
FRAME_SKIP = 1        # process every (FRAME_SKIP+1)th frame; 1 => process 1, skip 1
USE_V8N = True        # use nano model for speed

# tracking / association
PIXELS_PER_METER = 50.0
MAX_MISSED_FRAMES = 30
ASSIGNMENT_MAX_DISTANCE_PX = 180.0

# appearance (hist on torso ROI in HSV)
H_BINS, S_BINS = 10, 4
APPEARANCE_WEIGHT = 180.0
APPEARANCE_SMOOTHING = 0.2
MIN_HIST_AREA = 900

# color signature (HSV mean, not BGR)
COLOR_MATCH_WEIGHT = 60.0

# collision guard
COLLISION_FEET_DISTANCE_PX = 140.0
SWAP_DEBOUNCE_FRAMES = 8           # need N consecutive frames of better pairing to allow swap during collision
SWAP_MARGIN = 40.0                 # total-cost improvement required to consider swap in collision

PLAYER_NAMES = ("Player A", "Player B")
PLAYER_COLORS = ((0, 255, 0), (0, 165, 255))

COURT_WIDTH_METERS = 6.4
COURT_LENGTH_METERS = 9.75
CALIBRATION = {
    "enabled": False,
    "far_wall": {"y": None, "left": None, "right": None},
    "near_wall": {"y": None, "left": None, "right": None},
}

# -------------------- Kalman --------------------
class KalmanFilter2D:
    def __init__(self, process_noise: float = 1e-3, measurement_noise: float = 1e-1) -> None:
        self.dt = 1.0
        self.A = np.array([[1,0,self.dt,0],[0,1,0,self.dt],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 500.0
        self.Q = np.eye(4, dtype=np.float32) * process_noise
        self.R = np.eye(2, dtype=np.float32) * measurement_noise
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.initialized = False

    def _rebuild_transition(self, dt: float) -> None:
        self.A = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=np.float32)

    def reset(self) -> None:
        self.P = np.eye(4, dtype=np.float32) * 500.0
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.initialized = False

    def predict(self, dt: float) -> Optional[Tuple[float, float]]:
        if not self.initialized:
            return None
        if abs(dt - self.dt) > 1e-6:
            self.dt = dt
            self._rebuild_transition(dt)
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return float(self.x[0,0]), float(self.x[1,0])

    def correct(self, point: Tuple[float, float]) -> Tuple[float, float]:
        z = np.array([[point[0]],[point[1]]], dtype=np.float32)
        if not self.initialized:
            self.x[0,0] = point[0]
            self.x[1,0] = point[1]
            self.initialized = True
        else:
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            y = z - (self.H @ self.x)
            self.x = self.x + (K @ y)
            I = np.eye(self.P.shape[0], dtype=np.float32)
            self.P = (I - K @ self.H) @ self.P
        return float(self.x[0,0]), float(self.x[1,0])

# -------------------- Tracking --------------------
class PlayerTrack:
    def __init__(self, name: str, color: Tuple[int, int, int]) -> None:
        self.name = name
        self.color = color
        self.measurement_px: Optional[Tuple[float, float]] = None
        self.filtered_px: Optional[Tuple[float, float]] = None
        self.filtered_m: Optional[Tuple[float, float]] = None
        self.predicted_px: Optional[Tuple[float, float]] = None
        self.appearance_hist: Optional[np.ndarray] = None
        self.color_signature: Optional[np.ndarray] = None
        self.total_distance_pixels: float = 0.0
        self.total_distance_meters_accum: float = 0.0
        self.missed_frames: int = 0
        self.last_box: Optional[Tuple[int, int, int, int]] = None
        self.last_confidence: Optional[float] = None
        self.path: Deque[Tuple[int, int]] = deque(maxlen=120)
        self.filter = KalmanFilter2D()
        # collision swap protection
        self._swap_streak = 0

    def begin_frame(self, dt: float) -> None:
        prediction = self.filter.predict(dt)
        self.predicted_px = prediction if prediction is not None else self.filtered_px

    def update(self, measurement_px, box, confidence, hist, color_signature=None) -> None:
        prev_filtered_px = self.filtered_px
        prev_filtered_m = self.filtered_m
        prediction = self.predicted_px

        filtered_px = self.filter.correct(measurement_px)

        if prev_filtered_px is not None:
            self.total_distance_pixels += math.hypot(filtered_px[0]-prev_filtered_px[0],
                                                     filtered_px[1]-prev_filtered_px[1])
        elif prediction is not None:
            self.total_distance_pixels += math.hypot(filtered_px[0]-prediction[0],
                                                     filtered_px[1]-prediction[1])

        self.measurement_px = measurement_px
        self.filtered_px = filtered_px
        self.filtered_m = pixel_to_court_coords(*filtered_px)
        self.last_box = box
        self.last_confidence = confidence
        self.path.append((int(filtered_px[0]), int(filtered_px[1])))
        self.missed_frames = 0
        self.predicted_px = filtered_px
        self._update_appearance(hist)
        self._update_color(color_signature)
        self._swap_streak = 0  # successful update resets streak

        if self.filtered_m is not None:
            if prev_filtered_m is not None:
                self.total_distance_meters_accum += math.hypot(self.filtered_m[0]-prev_filtered_m[0],
                                                               self.filtered_m[1]-prev_filtered_m[1])
            elif prediction is not None:
                prev_m = pixel_to_court_coords(*prediction)
                if prev_m is not None:
                    self.total_distance_meters_accum += math.hypot(self.filtered_m[0]-prev_m[0],
                                                                   self.filtered_m[1]-prev_m[1])

    def _update_appearance(self, hist: Optional[np.ndarray]) -> None:
        if hist is None or hist.size == 0:
            return
        hist = cv2.normalize(hist, None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
        if self.appearance_hist is None:
            self.appearance_hist = hist
        else:
            blended = (1.0 - APPEARANCE_SMOOTHING) * self.appearance_hist + APPEARANCE_SMOOTHING * hist
            self.appearance_hist = cv2.normalize(blended, None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)

    def _update_color(self, color_signature: Optional[np.ndarray]) -> None:
        if color_signature is None:
            return
        if self.color_signature is None:
            self.color_signature = color_signature
        else:
            self.color_signature = (1.0 - APPEARANCE_SMOOTHING) * self.color_signature + APPEARANCE_SMOOTHING * color_signature

    def mark_missed(self) -> None:
        self.missed_frames += 1
        if self.predicted_px is not None:
            self.filtered_px = self.predicted_px
            self.filtered_m = pixel_to_court_coords(*self.predicted_px)
        if self.missed_frames > MAX_MISSED_FRAMES:
            self.reset()

    def bump_swap_streak(self):
        self._swap_streak += 1

    def reset_swap_streak(self):
        self._swap_streak = 0

    @property
    def swap_streak(self) -> int:
        return self._swap_streak

    def reset(self) -> None:
        self.measurement_px = None
        self.filtered_px = None
        self.filtered_m = None
        self.predicted_px = None
        self.appearance_hist = None
        self.color_signature = None
        self.total_distance_pixels = 0.0
        self.total_distance_meters_accum = 0.0
        self.missed_frames = 0
        self.last_box = None
        self.last_confidence = None
        self.path.clear()
        self.filter.reset()
        self._swap_streak = 0

    @property
    def total_distance_meters(self) -> float:
        if CALIBRATION["enabled"] and self.total_distance_meters_accum > 0.0:
            return self.total_distance_meters_accum
        return 0.0 if PIXELS_PER_METER <= 0 else self.total_distance_pixels / PIXELS_PER_METER

players: List[PlayerTrack] = [PlayerTrack(n, c) for n, c in zip(PLAYER_NAMES, PLAYER_COLORS)]

# -------------------- Utils --------------------
def try_set_process_dpi_aware() -> None:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except AttributeError:
        pass

def get_window_title(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    if length == 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value.strip()

def get_window_rect(hwnd: int) -> Optional[Dict[str, int]]:
    rect = wintypes.RECT()
    if not user32.IsWindow(hwnd):
        return None
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    width = rect.right - rect.left
    height = rect.bottom - rect.top
    if width <= 0 or height <= 0:
        return None
    return {"top": rect.top, "left": rect.left, "width": width, "height": height}

def wait_for_window_selection() -> int:
    try_set_process_dpi_aware()
    print("Click on the window you want to monitor, then release the mouse button...")
    lb_was_down = bool(user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000)
    while True:
        time.sleep(0.01)
        lb_down = bool(user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000)
        if lb_down and not lb_was_down:
            cursor = wintypes.POINT()
            user32.GetCursorPos(ctypes.byref(cursor))
            hwnd = user32.WindowFromPoint(cursor)
            if not hwnd:
                print("No window detected. Click again.")
            else:
                hwnd = user32.GetAncestor(hwnd, GA_ROOT)
                rect = get_window_rect(hwnd)
                if rect:
                    title = get_window_title(hwnd) or "Untitled window"
                    print(f"Selected window: {title} ({rect['width']}x{rect['height']})")
                    return hwnd
                print("Could not read window bounds. Try another window.")
        lb_was_down = lb_down

def calibration_ready() -> bool:
    far = CALIBRATION["far_wall"]; near = CALIBRATION["near_wall"]
    required = (far["y"], far["left"], far["right"], near["y"], near["left"], near["right"])
    return all(value is not None for value in required)

def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def depth_alpha(y: float) -> float:
    far_y = CALIBRATION["far_wall"]["y"]; near_y = CALIBRATION["near_wall"]["y"]
    if far_y is None or near_y is None or near_y == far_y:
        return 0.0
    t = (y - far_y) / (near_y - far_y)
    return clamp(t, 0.0, 1.0)

def pixel_to_court_coords(x: float, y: float) -> Optional[Tuple[float, float]]:
    if not CALIBRATION["enabled"]:
        return None
    if not calibration_ready():
        raise ValueError("Calibration enabled but pixel references incomplete.")
    far = CALIBRATION["far_wall"]; near = CALIBRATION["near_wall"]
    alpha = depth_alpha(y)
    left_boundary = lerp(far["left"], near["left"], alpha)
    right_boundary = lerp(far["right"], near["right"], alpha)
    width_pixels = right_boundary - left_boundary
    if width_pixels <= 0:
        return None
    x_meters = (x - left_boundary) * (COURT_WIDTH_METERS / width_pixels)
    y_meters = alpha * COURT_LENGTH_METERS
    return x_meters, y_meters

def compute_feet_point(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, _, x2, y2 = box
    return (x1 + x2) / 2.0, float(y2)

def clip_box(box, width, height) -> Optional[Tuple[int, int, int, int]]:
    x1,y1,x2,y2 = box
    x1 = max(0, min(x1, width - 1)); x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1)); y2 = max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1,y1,x2,y2

def _torso_roi(box: Tuple[int,int,int,int]) -> Tuple[int,int,int,int]:
    """Upper-middle region of the box (shirt area), reduces background."""
    x1,y1,x2,y2 = box
    h = y2 - y1; w = x2 - x1
    top = y1 + int(0.15*h)
    bottom = y1 + int(0.55*h)
    left = x1 + int(0.20*w)
    right = x1 + int(0.80*w)
    return left, top, right, bottom

def compute_histogram_torso_hs(frame: np.ndarray, box: Optional[Tuple[int, int, int, int]]) -> Optional[np.ndarray]:
    if box is None: return None
    rx1, ry1, rx2, ry2 = _torso_roi(box)
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    roi = frame[ry1:ry2, rx1:rx2]
    if roi.size == 0 or (roi.shape[0] * roi.shape[1]) < MIN_HIST_AREA:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1], None, [H_BINS, S_BINS], [0,180, 0,256])
    hist = cv2.normalize(hist, None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return hist.flatten().astype(np.float32)

def compute_color_signature_hsv(frame: np.ndarray, box: Optional[Tuple[int,int,int,int]]) -> Optional[np.ndarray]:
    if box is None: return None
    rx1, ry1, rx2, ry2 = _torso_roi(box)
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    roi = frame[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mean = cv2.mean(hsv)[:3]
    return np.array(mean, dtype=np.float32)

def _hist_dist(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.5  # neutral distance
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))

def association_cost(track, detection) -> float:
    expected = track.predicted_px or track.filtered_px
    if expected is None:
        return float("inf")
    pos = detection["feet_px"]
    dist = math.hypot(pos[0]-expected[0], pos[1]-expected[1])
    if dist > ASSIGNMENT_MAX_DISTANCE_PX:
        return float("inf")
    cost = dist
    # appearance distances
    cost += APPEARANCE_WEIGHT * _hist_dist(track.appearance_hist, detection["hist"])
    if track.color_signature is not None and detection.get("color_signature") is not None:
        color_dist = np.linalg.norm(track.color_signature - detection["color_signature"]) / 255.0
        cost += COLOR_MATCH_WEIGHT * color_dist
    return cost

def _feet_distance_px(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def associate_detections(tracks: List[PlayerTrack], detections: List[Dict[str, object]]) -> None:
    # mark missed if no detections
    if not detections:
        for t in tracks: t.mark_missed()
        return

    remaining = detections.copy()
    assigned: List[PlayerTrack] = []

    # Bootstrap with left/right if uninitialized
    uninitialized = [t for t in tracks if t.filtered_px is None and t.predicted_px is None]
    if uninitialized:
        remaining.sort(key=lambda d: d["feet_px"][0])
        for t, det in zip(uninitialized, remaining):
            t.update(det["feet_px"], det["box"], det["confidence"], det["hist"], det.get("color_signature"))
            assigned.append(t)
        remaining = remaining[len(uninitialized):]

    active = [t for t in tracks if t not in assigned]
    if active and remaining:
        remaining.sort(key=lambda d: d["confidence"], reverse=True)
        candidate = remaining[:len(active)]

        if len(active) == 1:
            t = active[0]; d = candidate[0]
            c = association_cost(t, d)
            if math.isfinite(c):
                t.update(d["feet_px"], d["box"], d["confidence"], d["hist"], d.get("color_signature"))
                assigned.append(t); remaining.remove(d)
        elif len(active) >= 2 and len(candidate) >= 2:
            ta, tb = active[:2]; da, db = candidate[:2]
            # compute both pairings
            c_aa = association_cost(ta, da); c_bb = association_cost(tb, db)
            c_ab = association_cost(ta, db); c_ba = association_cost(tb, da)

            best_pairs = None
            best_cost = float("inf")

            # default: keep previous ID mapping
            keep_pairs = [(ta, da), (tb, db)]
            keep_total = (c_aa if math.isfinite(c_aa) else 1e9) + (c_bb if math.isfinite(c_bb) else 1e9)

            swap_pairs = [(ta, db), (tb, da)]
            swap_total = (c_ab if math.isfinite(c_ab) else 1e9) + (c_ba if math.isfinite(c_ba) else 1e9)

            # Collision guard: if feet of detections are very close -> prefer KEEP unless SWAP is much better for several frames
            feet_dist = _feet_distance_px(da["feet_px"], db["feet_px"])
            collision = feet_dist < COLLISION_FEET_DISTANCE_PX

            if collision:
                # If swap is clearly better, increment streak; otherwise reset
                if swap_total + SWAP_MARGIN < keep_total:
                    # bump streak on both tracks (they share the event)
                    ta.bump_swap_streak(); tb.bump_swap_streak()
                    if min(ta.swap_streak, tb.swap_streak) >= SWAP_DEBOUNCE_FRAMES:
                        best_pairs, best_cost = swap_pairs, swap_total
                    else:
                        best_pairs, best_cost = keep_pairs, keep_total
                else:
                    ta.reset_swap_streak(); tb.reset_swap_streak()
                    best_pairs, best_cost = keep_pairs, keep_total
            else:
                # No collision → pick lower cost and reset streaks
                ta.reset_swap_streak(); tb.reset_swap_streak()
                if swap_total < keep_total:
                    best_pairs, best_cost = swap_pairs, swap_total
                else:
                    best_pairs, best_cost = keep_pairs, keep_total

            for t, d in best_pairs:
                t.update(d["feet_px"], d["box"], d["confidence"], d["hist"], d.get("color_signature"))
                assigned.append(t)
                if d in remaining: remaining.remove(d)

        # any leftover tracks
        leftover = [t for t in active if t not in assigned]
        for t in leftover:
            best_d, best_c = None, float("inf")
            for d in remaining:
                c = association_cost(t, d)
                if math.isfinite(c) and c < best_c:
                    best_c, best_d = c, d
            if best_d is not None:
                t.update(best_d["feet_px"], best_d["box"], best_d["confidence"], best_d["hist"], best_d.get("color_signature"))
                assigned.append(t); remaining.remove(best_d)

    for t in tracks:
        if t not in assigned:
            t.mark_missed()

def reset_players_state() -> None:
    for t in players: t.reset()

# -------------------- Main --------------------
def main() -> None:
    reset_players_state()
    model = YOLO("yolov8n.pt" if USE_V8N else "yolov8s.pt")
    sct = mss.mss()
    hwnd = wait_for_window_selection()

    cv2.namedWindow("YOLO Desktop Detection", cv2.WINDOW_NORMAL)
    last_time = time.time()
    frame_idx = 0

    while True:
        now = time.time()
        dt = max(now - last_time, 1e-3)
        last_time = now

        region = get_window_rect(hwnd)
        if region is None:
            print("Selected window is no longer available. Exiting.")
            break

        for t in players:
            t.begin_frame(dt)

        img = np.array(sct.grab(region))
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        h, w = frame_bgr.shape[:2]
        annotated = frame_bgr.copy()

        # Frame skip for speed
        run_yolo = (frame_idx % (FRAME_SKIP + 1) == 0)
        detections: List[Dict[str, object]] = []

        if run_yolo:
            # NOTE: Pass BGR numpy array directly; restrict to persons and target input size
            results = model(frame_bgr, classes=[PERSON_CLASS_ID], imgsz=IMGSZ, verbose=False)
            for r in results:
                boxes = r.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                xyxy = boxes.xyxy.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)

                for (x1,y1,x2,y2), conf, cls in zip(xyxy, confs, classes):
                    if conf < CONF_THRESHOLD or cls != PERSON_CLASS_ID:
                        continue
                    box = clip_box((x1,y1,x2,y2), w, h)
                    if box is None: continue
                    feet = compute_feet_point(box)

                    # torso HSV hist + HSV mean as color signature
                    hist = compute_histogram_torso_hs(frame_bgr, box)
                    color_sig = compute_color_signature_hsv(frame_bgr, box)

                    detections.append({
                        "box": box,
                        "confidence": float(conf),
                        "feet_px": feet,
                        "hist": hist,
                        "color_signature": color_sig,
                    })

            associate_detections(players, detections)

        # draw overlays
        for t in players:
            if t.last_box is None:
                continue
            x1,y1,x2,y2 = t.last_box
            cv2.rectangle(annotated, (x1,y1), (x2,y2), t.color, 2)
            dist_m = t.total_distance_meters
            conf = t.last_confidence if t.last_confidence is not None else 0.0
            text_y = min(y2 + 20, h - 20)
            label = f"{t.name} {conf:.2f} ({dist_m:.1f} m)"
            cv2.putText(annotated, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, t.color, 2)
            for i in range(1, len(t.path)):
                cv2.line(annotated, t.path[i-1], t.path[i], t.color, 2)

        y = 60
        for t in players:
            summary = f"{t.name}: {t.total_distance_meters:.1f} m"
            cv2.putText(annotated, summary, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, t.color, 2)
            y += 25

        cv2.imshow("YOLO Desktop Detection", annotated)
        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("r"), ord("R")):
            reset_players_state()
            last_time = time.time()
            frame_idx = 0
            continue
        if key in (27, ord("q"), ord("Q")):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
