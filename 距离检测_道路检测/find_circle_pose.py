"""模型3矩形中心检测：OpenCV 主检，YOLO11 仅在连续漏检时复核。"""

import math
import os

import cv2
import numpy as np

from maix import app, camera, display, image, nn, time


CAM_W = 640
CAM_H = 480
CAM_FPS = 60
CAM_BUFF_NUM = 3
CAMERA_WARMUP_FRAMES = 20

PROC_W = 320
PROC_H = 240

MIN_QUAD_AREA = 3500
MAX_QUAD_AREA = 70000
EDGE_MARGIN = 4
MIN_EDGE_LENGTH_SQ = 30 * 30
MAX_CORNER_COS_SQ = 60
MAX_OPPOSITE_EDGE_RATIO = 4
MIN_BBOX_FILL_PERCENT = 50
APPROX_RATIO = 0.025

MISS_TRIGGER_FRAMES = 2
YOLO_COOLDOWN_FRAMES = 5
YOLO_CONFIDENCE = 0.25
YOLO_IOU = 0.45
YOLO_ROI_PADDING_RATIO = 0.10

STATUS_INTERVAL_MS = 1000
MODEL_EVAL_INTERVAL_MS = 10000
MODEL_DISABLE_INVALID_RATIO = 0.20
MODEL_DISABLE_MIN_ATTEMPTS = 5
MODEL_DISABLE_MAX_RESCUE_RATIO = 0.10


# =========================
# 25.5 cm x 17 cm A4 横向矩形靶测距参数
# =========================
# 单位统一使用 mm，避免 cm / mm 混用。
TARGET_W_MM = 255.0
TARGET_H_MM = 170.0

# 物理角点顺序必须与图像角点顺序一致：左上、右上、右下、左下。
OBJECT_POINTS = np.array(
    [
        [-TARGET_W_MM / 2, -TARGET_H_MM / 2, 0.0],  # 左上
        [ TARGET_W_MM / 2, -TARGET_H_MM / 2, 0.0],  # 右上
        [ TARGET_W_MM / 2,  TARGET_H_MM / 2, 0.0],  # 右下
        [-TARGET_W_MM / 2,  TARGET_H_MM / 2, 0.0],  # 左下
    ],
    dtype=np.float32,
)

# MaixCAM2 640x480 内参。
# 运行 calibrate_camera.py 标定后，把输出的 fx/fy 填到这里。
CAMERA_FX = 508.0
CAMERA_FY = 504.0
CAMERA_CX = CAM_W / 2.0
CAMERA_CY = CAM_H / 2.0

CAMERA_MATRIX = np.array(
    [
        [CAMERA_FX, 0.0, CAMERA_CX],
        [0.0, CAMERA_FY, CAMERA_CY],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# 没有正式标定前，先假设镜头畸变为 0。
# 如果你后续做了棋盘格标定，把真实畸变参数替换这里。
DIST_COEFFS = np.zeros((5, 1), dtype=np.float64)

# 距离输出和滤波参数。
POSE_PRINT_INTERVAL_MS = 200
POSE_FILTER_ALPHA = 0.35


def polygon_area(points):
    area2 = 0
    for index in range(4):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % 4]
        area2 += x1 * y2 - x2 * y1
    return abs(area2) // 2


def bbox_from_points(points):
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def edge_length_sq(first, second):
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    return dx * dx + dy * dy


def is_convex_quad(points):
    sign = 0
    for index in range(4):
        first = points[index]
        second = points[(index + 1) % 4]
        third = points[(index + 2) % 4]
        cross = (
            (second[0] - first[0]) * (third[1] - second[1])
            - (second[1] - first[1]) * (third[0] - second[0])
        )
        if cross == 0:
            return False
        current_sign = 1 if cross > 0 else -1
        if sign and current_sign != sign:
            return False
        sign = current_sign
    return True


def has_rectangle_geometry(points):
    lengths = []
    for index in range(4):
        previous = points[(index - 1) % 4]
        current = points[index]
        following = points[(index + 1) % 4]

        ax = previous[0] - current[0]
        ay = previous[1] - current[1]
        bx = following[0] - current[0]
        by = following[1] - current[1]
        length_a = ax * ax + ay * ay
        length_b = bx * bx + by * by

        if length_a < MIN_EDGE_LENGTH_SQ or length_b < MIN_EDGE_LENGTH_SQ:
            return False

        dot = ax * bx + ay * by
        if dot * dot * 100 > MAX_CORNER_COS_SQ * length_a * length_b:
            return False

        lengths.append(edge_length_sq(current, following))

    if max(lengths[0], lengths[2]) > (
        min(lengths[0], lengths[2]) * MAX_OPPOSITE_EDGE_RATIO
    ):
        return False
    if max(lengths[1], lengths[3]) > (
        min(lengths[1], lengths[3]) * MAX_OPPOSITE_EDGE_RATIO
    ):
        return False
    return True


def quad_is_target(points, frame_w=PROC_W, frame_h=PROC_H):
    if len(points) != 4 or not is_convex_quad(points):
        return False

    x, y, width, height = bbox_from_points(points)
    if width <= 0 or height <= 0:
        return False
    if x <= EDGE_MARGIN or y <= EDGE_MARGIN:
        return False
    if x + width >= frame_w - EDGE_MARGIN:
        return False
    if y + height >= frame_h - EDGE_MARGIN:
        return False

    area = polygon_area(points)
    if area < MIN_QUAD_AREA or area > MAX_QUAD_AREA:
        return False
    if area * 100 < width * height * MIN_BBOX_FILL_PERCENT:
        return False
    return has_rectangle_geometry(points)


def diagonal_intersection(points, frame_w=CAM_W, frame_h=CAM_H):
    x1, y1 = points[0]
    x2, y2 = points[2]
    x3, y3 = points[1]
    x4, y4 = points[3]

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        return None

    first_cross = x1 * y2 - y1 * x2
    second_cross = x3 * y4 - y3 * x4
    center_x = (
        first_cross * (x3 - x4) - (x1 - x2) * second_cross
    ) / denominator
    center_y = (
        first_cross * (y3 - y4) - (y1 - y2) * second_cross
    ) / denominator

    center = [int(round(center_x)), int(round(center_y))]
    if not (0 <= center[0] < frame_w and 0 <= center[1] < frame_h):
        return None
    return center


def contour_to_points(approx):
    flat = approx.reshape(-1, 2)
    return [(int(point[0]), int(point[1])) for point in flat]


def select_quad_from_binary(binary):
    contour_result = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = contour_result[-2]
    best = None
    best_area = 0

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < MIN_QUAD_AREA or contour_area > MAX_QUAD_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(contour, APPROX_RATIO * perimeter, True)
        if len(approx) != 4:
            continue

        points = contour_to_points(approx)
        if not quad_is_target(points):
            continue

        area = polygon_area(points)
        if area > best_area:
            best = points
            best_area = area

    return best


def border_is_white(binary):
    frame_h, frame_w = binary.shape[:2]
    margin = 2
    samples = (
        (margin, margin),
        (frame_w - margin - 1, margin),
        (margin, frame_h - margin - 1),
        (frame_w - margin - 1, frame_h - margin - 1),
    )
    white_count = 0
    for x, y in samples:
        if int(binary[y, x]) != 0:
            white_count += 1
    return white_count >= 3


def detect_rectangle(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # ---- 路径 1: 固定阈值（适配打印黑条纹边框 L:29~43 → gray:74~110）----
    # 取略高于 L=43 对应灰度值作为阈值，保证边框像素被捕获。
    _, fixed_binary = cv2.threshold(
        gray, 120, 255, cv2.THRESH_BINARY_INV,
    )
    if border_is_white(fixed_binary):
        fixed_binary = cv2.bitwise_not(fixed_binary)
    result = select_quad_from_binary(fixed_binary)
    if result is not None:
        return result

    # ---- 路径 2: Otsu 全局阈值（适合中近距离、目标占比大）----
    otsu_level, normal_binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    if border_is_white(normal_binary):
        _, otsu_binary = cv2.threshold(
            gray, otsu_level, 255, cv2.THRESH_BINARY_INV
        )
    else:
        otsu_binary = normal_binary
    result = select_quad_from_binary(otsu_binary)
    if result is not None:
        return result

    # ---- 路径 3: 自适应阈值回退（适配远距离、低对比度场景）----
    # C 从 9 降至 5：打印边框对比度比电工胶带低，需要更小的常数减量。
    adaptive_binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21,   # blockSize
        3,    # C — 降低以适配淡色边框
    )
    if border_is_white(adaptive_binary):
        adaptive_binary = cv2.bitwise_not(adaptive_binary)
    return select_quad_from_binary(adaptive_binary)


def scale_points(points, src_w, src_h, dst_w, dst_h):
    if points is None:
        return None

    scale_x = (dst_w - 1) / max(1, src_w - 1)
    scale_y = (dst_h - 1) / max(1, src_h - 1)
    return [
        (
            max(0, min(dst_w - 1, int(round(x * scale_x)))),
            max(0, min(dst_h - 1, int(round(y * scale_y)))),
        )
        for x, y in points
    ]




def distance_2d(first, second):
    dx = float(second[0]) - float(first[0])
    dy = float(second[1]) - float(first[1])
    return math.sqrt(dx * dx + dy * dy)


def order_quad_points(points):
    """把任意四角点排序为：左上、右上、右下、左下。"""
    if points is None or len(points) != 4:
        return None

    pts = np.array(points, dtype=np.float32)
    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 0] - pts[:, 1]

    top_left = pts[np.argmin(sums)]
    bottom_right = pts[np.argmax(sums)]
    top_right = pts[np.argmax(diffs)]
    bottom_left = pts[np.argmin(diffs)]

    return np.array(
        [
            top_left,
            top_right,
            bottom_right,
            bottom_left,
        ],
        dtype=np.float32,
    )


def target_pixel_width(ordered_points):
    """计算横向靶宽在图像中的平均像素宽度。"""
    if ordered_points is None or len(ordered_points) != 4:
        return 0.0

    top_width = distance_2d(ordered_points[0], ordered_points[1])
    bottom_width = distance_2d(ordered_points[3], ordered_points[2])
    return (top_width + bottom_width) * 0.5


def estimate_target_pose(points):
    """
    根据 70 cm x 60 cm 矩形靶四角点估算目标位姿。

    返回单位：
    - x_mm: 靶心相对摄像头左右偏移，左负右正
    - y_mm: 靶心相对摄像头上下偏移，上负下正
    - z_mm: 沿摄像头光轴的前方距离
    - distance_mm: 摄像头到靶心的空间直线距离
    - angle_x_deg: 水平角，左负右正
    - simple_z_mm: 用宽度反推的简易距离，用于对照
    """
    image_points = order_quad_points(points)
    if image_points is None:
        return None

    pixel_width = target_pixel_width(image_points)
    simple_z_mm = None
    if pixel_width > 1.0:
        simple_z_mm = CAMERA_FX * TARGET_W_MM / pixel_width

    # 长方形平面目标优先使用 IPPE；如果当前 OpenCV 不支持，就退回 ITERATIVE。
    pnp_flag = getattr(cv2, "SOLVEPNP_IPPE", getattr(cv2, "SOLVEPNP_ITERATIVE", 0))

    try:
        success, rvec, tvec = cv2.solvePnP(
            OBJECT_POINTS,
            image_points,
            CAMERA_MATRIX,
            DIST_COEFFS,
            flags=pnp_flag,
        )
    except Exception as err:
        print("[pose] solvePnP failed: {}".format(err))
        if simple_z_mm is None:
            return None
        return {
            "x_mm": 0.0,
            "y_mm": 0.0,
            "z_mm": simple_z_mm,
            "distance_mm": simple_z_mm,
            "angle_x_deg": 0.0,
            "simple_z_mm": simple_z_mm,
            "rvec": None,
            "tvec": None,
            "image_points": image_points,
        }

    if not success:
        if simple_z_mm is None:
            return None
        return {
            "x_mm": 0.0,
            "y_mm": 0.0,
            "z_mm": simple_z_mm,
            "distance_mm": simple_z_mm,
            "angle_x_deg": 0.0,
            "simple_z_mm": simple_z_mm,
            "rvec": None,
            "tvec": None,
            "image_points": image_points,
        }

    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
    x_mm = float(tvec[0])
    y_mm = float(tvec[1])
    z_mm = float(tvec[2])

    if z_mm <= 0:
        return None

    distance_mm = math.sqrt(x_mm * x_mm + y_mm * y_mm + z_mm * z_mm)
    angle_x_deg = math.degrees(math.atan2(x_mm, z_mm))

    return {
        "x_mm": x_mm,
        "y_mm": y_mm,
        "z_mm": z_mm,
        "distance_mm": distance_mm,
        "angle_x_deg": angle_x_deg,
        "simple_z_mm": simple_z_mm,
        "rvec": rvec,
        "tvec": tvec,
        "image_points": image_points,
    }


# =========================
# A4 纸内部几何图形测量
# =========================
# 透视校正后测量 A4 纸内部的圆（直径）和矩形（边长），单位 mm。

def measure_internal_shapes(frame_bgr, quad_points, pose=None):
    """在 A4 纸 bounding box 内检测黑色图形（矩形/三角形），返回像素坐标和估算物理尺寸。

    流程：
    1. 用 A4 四角点算出外接矩形，向内缩一点排除外边框
    2. 截取 ROI → 灰度 → 固定阈值二值化
    3. 找轮廓 → 多边形逼近 → 只保留三角形和四边形
    4. 用相似三角形公式 (sw * D) / fx 估算物理宽度
    """
    if quad_points is None or len(quad_points) != 4:
        return None

    # 1. 计算 bounding box，向内收缩以排除 A4 纸外边框
    xs = [p[0] for p in quad_points]
    ys = [p[1] for p in quad_points]
    bx = int(min(xs))
    by = int(min(ys))
    bw = int(max(xs) - bx)
    bh = int(max(ys) - by)

    BORDER_SHRINK = 12  # 向内缩 12px，跳过外边框
    bx += BORDER_SHRINK
    by += BORDER_SHRINK
    bw -= 2 * BORDER_SHRINK
    bh -= 2 * BORDER_SHRINK

    if bw <= 20 or bh <= 20:
        return None

    # 2. 截取 ROI 并二值化
    roi = frame_bgr[by:by + bh, bx:bx + bw]
    if roi.size == 0:
        return None

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi_bin = cv2.threshold(roi_gray, 100, 255, cv2.THRESH_BINARY_INV)

    contours_s, _ = cv2.findContours(
        roi_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    shapes = []
    for sc in contours_s:
        s_area = cv2.contourArea(sc)
        if s_area < 200:
            continue

        s_peri = cv2.arcLength(sc, True)
        if s_peri <= 0:
            continue

        s_approx = cv2.approxPolyDP(sc, 0.02 * s_peri, True)
        n_vertices = len(s_approx)

        sx, sy, sw, sh = cv2.boundingRect(sc)
        sx_orig = bx + sx
        sy_orig = by + sy

        # ---- 形状分类 ----
        if n_vertices == 3:
            # 三角形：提取 3 个顶点，亚像素精化，计算三条边长
            tri_pts = s_approx.reshape(3, 2).astype(np.float32)

            # 亚像素角点精化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        20, 0.001)
            cv2.cornerSubPix(roi_gray, tri_pts, (3, 3), (-1, -1), criteria)

            # 计算三条边长 (px)，考虑 fx/fy 差异
            sides_px = []
            sides_mm = []
            for i in range(3):
                p1 = tri_pts[i]
                p2 = tri_pts[(i + 1) % 3]
                dx = float(p2[0] - p1[0])
                dy = float(p2[1] - p1[1])
                side_px = math.hypot(dx, dy)
                sides_px.append(round(side_px, 1))

                # 物理边长：考虑 fx/fy 差异
                if pose is not None and pose.get("z_mm", 0) > 0:
                    z = pose["z_mm"]
                    side_mm = math.hypot(
                        dx * z / CAMERA_FX,
                        dy * z / CAMERA_FY,
                    )
                    sides_mm.append(round(side_mm, 1))

            # 顶点映射到原始图像坐标
            tri_pts_orig = [
                (bx + int(round(p[0])), by + int(round(p[1])))
                for p in tri_pts
            ]

            shapes.append({
                "type": "tri",
                "vertices": tri_pts_orig,
                "sides_px": sides_px,
                "sides_mm": sides_mm if sides_mm else None,
                "bbox": (sx_orig, sy_orig, sw, sh),
            })

        elif n_vertices == 4:
            # 矩形：用 bounding box 宽高估算
            width_mm = None
            height_mm = None
            if pose is not None and pose.get("z_mm", 0) > 0:
                width_mm = (sw * pose["z_mm"]) / CAMERA_FX
                height_mm = (sh * pose["z_mm"]) / CAMERA_FY

            shapes.append({
                "type": "rect",
                "bbox": (sx_orig, sy_orig, sw, sh),
                "width_mm": width_mm,
                "height_mm": height_mm,
            })

        else:
            # ---- 圆形度判断：4π·area / perimeter²，正圆=1.0 ----
            circularity = (4.0 * math.pi * s_area) / (s_peri * s_peri)
            if circularity < 0.78:
                continue  # 既不是多边形也不是圆，跳过

            (cx, cy), radius = cv2.minEnclosingCircle(sc)
            cx_orig = bx + int(cx)
            cy_orig = by + int(cy)
            r_px = int(radius)
            diameter_mm = None
            if pose is not None and pose.get("z_mm", 0) > 0:
                diameter_mm = (2.0 * r_px * pose["z_mm"]) / CAMERA_FX

            shapes.append({
                "type": "circle",
                "center": (cx_orig, cy_orig),
                "radius_px": r_px,
                "diameter_mm": diameter_mm,
                "bbox": (sx_orig, sy_orig, sw, sh),
            })

    return {"shapes": shapes} if shapes else None


def update_rescue_schedule(miss_streak, cooldown, opencv_valid, yolo_enabled):
    """推进一次复核状态；冷却中的完整采集帧不能触发 YOLO。"""
    was_cooling = cooldown > 0
    if was_cooling:
        cooldown -= 1

    if opencv_valid:
        miss_streak = 0
    else:
        miss_streak += 1

    should_try = (
        not opencv_valid
        and yolo_enabled
        and not was_cooling
        and miss_streak >= MISS_TRIGGER_FRAMES
    )
    return miss_streak, cooldown, should_try


def should_disable_yolo(frame_count, invalid_count, attempts, rescues):
    """判断当前评估窗口内 YOLO 复核是否没有净收益。"""
    invalid_ratio = invalid_count / max(1, frame_count)
    rescue_ratio = rescues / max(1, attempts)
    return (
        invalid_ratio > MODEL_DISABLE_INVALID_RATIO
        and attempts >= MODEL_DISABLE_MIN_ATTEMPTS
        and rescue_ratio < MODEL_DISABLE_MAX_RESCUE_RATIO
    )


class FindRectCircle:
    """保持原五项返回接口的矩形中心检测器。"""

    DEBUG = False
    PRINT_TIME = False
    debug_draw_err_line = True
    debug_draw_err_msg = False
    debug_draw_circle = False
    debug_draw_rect = True
    debug_show_hires = False

    # 兼容旧移植契约；新的主路径固定使用 640x480 OpenCV 图像。
    hires_mode = False
    model_path = "/root/models/_model25e_maixcam2/best.mud"
    model_dual_buff_mode = False

    def __init__(self, disp):
        model_path = self.model_path
        if not os.path.exists(model_path):
            local_model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "best.mud",
            )
            if not os.path.exists(local_model_path):
                print(
                    "load model failed, please put model in {}, or {}".format(
                        self.model_path, local_model_path
                    )
                )
            model_path = local_model_path

        self.disp = disp
        self.detector = nn.YOLO11(model=model_path, dual_buff=self.model_dual_buff_mode)
        self.legacy_width = self.detector.input_width()
        self.legacy_height = self.detector.input_height()

        self.cam = camera.Camera(
            CAM_W,
            CAM_H,
            image.Format.FMT_BGR888,
            fps=CAM_FPS,
            buff_num=CAM_BUFF_NUM,
        )
        self.cam.skip_frames(CAMERA_WARMUP_FRAMES)

        self.center_pos = [self.legacy_width // 2, self.legacy_height // 2]
        self.last_center = list(self.center_pos)
        self.err_center = [0, 0]
        self.last_circle3_points = []
        self.updated = False

        # 最近一次有效测距结果；run() 保持原五项返回接口，外部可通过 get_pose() 读取。
        self.last_pose = None
        self.filtered_pose = None
        self.pose_print_tick = 0

        self.miss_streak = 0
        self.yolo_cooldown = 0
        self.yolo_enabled = True

        now_ms = time.ticks_ms()
        self.status_tick = now_ms
        self.status_frames = 0
        self.status_opencv_valid = 0
        self.status_yolo_attempts = 0
        self.status_yolo_rescues = 0

        self.model_eval_tick = now_ms
        self.eval_frames = 0
        self.eval_opencv_invalid = 0
        self.eval_yolo_attempts = 0
        self.eval_yolo_rescues = 0

    def get_res(self):
        return [self.legacy_width, self.legacy_height]

    def get_pose(self):
        """返回最近一次滤波后的测距结果，单位为 mm / deg；无有效目标时为 None。"""
        return self.last_pose

    def _to_legacy_center(self, center):
        scale_x = (self.legacy_width - 1) / max(1, CAM_W - 1)
        scale_y = (self.legacy_height - 1) / max(1, CAM_H - 1)
        return [
            int(round(center[0] * scale_x)),
            int(round(center[1] * scale_y)),
        ]

    def _to_legacy_error(self, center):
        legacy_center = self._to_legacy_center(center)
        return [
            legacy_center[0] - self.center_pos[0],
            legacy_center[1] - self.center_pos[1],
        ]

    def _try_yolo_rescue(self, img, frame_bgr):
        input_w = self.detector.input_width()
        input_h = self.detector.input_height()
        img_ai = img.resize(input_w, input_h)
        img_ai = img_ai.to_format(self.detector.input_format())

        objs = self.detector.detect(
            img_ai,
            conf_th=YOLO_CONFIDENCE,
            iou_th=YOLO_IOU,
        )
        scale_x = CAM_W / input_w
        scale_y = CAM_H / input_h
        best_points = None
        best_area = 0

        for obj in objs:
            roi_x = int(round(obj.x * scale_x))
            roi_y = int(round(obj.y * scale_y))
            roi_w = max(1, int(round(obj.w * scale_x)))
            roi_h = max(1, int(round(obj.h * scale_y)))
            pad_x = max(1, int(round(roi_w * YOLO_ROI_PADDING_RATIO)))
            pad_y = max(1, int(round(roi_h * YOLO_ROI_PADDING_RATIO)))

            x1 = max(0, roi_x - pad_x)
            y1 = max(0, roi_y - pad_y)
            x2 = min(CAM_W, roi_x + roi_w + pad_x)
            y2 = min(CAM_H, roi_y + roi_h + pad_y)
            roi_width = x2 - x1
            roi_height = y2 - y1
            if roi_width < 2 or roi_height < 2:
                continue

            roi_bgr = frame_bgr[y1:y2, x1:x2]
            resized_roi = cv2.resize(
                roi_bgr,
                (PROC_W, PROC_H),
                interpolation=cv2.INTER_AREA,
            )
            rescue_points = detect_rectangle(resized_roi)
            if rescue_points is None:
                continue

            mapped = scale_points(
                rescue_points,
                PROC_W,
                PROC_H,
                roi_width,
                roi_height,
            )
            mapped = [(x + x1, y + y1) for x, y in mapped]
            area = polygon_area(mapped)
            if area > best_area:
                best_points = mapped
                best_area = area

        return best_points

    def _update_model_policy(self, now_ms):
        elapsed_ms = (now_ms - self.model_eval_tick) & 0xFFFFFFFF
        if elapsed_ms < MODEL_EVAL_INTERVAL_MS:
            return

        invalid_ratio = self.eval_opencv_invalid / max(1, self.eval_frames)
        rescue_ratio = self.eval_yolo_rescues / max(1, self.eval_yolo_attempts)
        if self.yolo_enabled and should_disable_yolo(
            self.eval_frames,
            self.eval_opencv_invalid,
            self.eval_yolo_attempts,
            self.eval_yolo_rescues,
        ):
            self.yolo_enabled = False
            print(
                "[vision] YOLO rescue disabled: invalid={:.1f}% rescue={:.1f}%".format(
                    invalid_ratio * 100.0,
                    rescue_ratio * 100.0,
                )
            )

        self.model_eval_tick = now_ms
        self.eval_frames = 0
        self.eval_opencv_invalid = 0
        self.eval_yolo_attempts = 0
        self.eval_yolo_rescues = 0

    def _report_status(self, now_ms):
        elapsed_ms = (now_ms - self.status_tick) & 0xFFFFFFFF
        if elapsed_ms < STATUS_INTERVAL_MS:
            return

        fps = self.status_frames * 1000.0 / max(1, elapsed_ms)
        valid_ratio = self.status_opencv_valid * 100.0 / max(
            1, self.status_frames
        )
        print(
            "[vision] fps={:.1f} opencv_valid={:.1f}% "
            "miss={} yolo={}/{} enabled={}".format(
                fps,
                valid_ratio,
                self.miss_streak,
                self.status_yolo_attempts,
                self.status_yolo_rescues,
                int(self.yolo_enabled),
            )
        )

        self.status_tick = now_ms
        self.status_frames = 0
        self.status_opencv_valid = 0
        self.status_yolo_attempts = 0
        self.status_yolo_rescues = 0

    def _update_pose_filter(self, pose):
        if pose is None:
            self.last_pose = None
            return None

        keys = ("x_mm", "y_mm", "z_mm", "distance_mm", "angle_x_deg")
        if self.filtered_pose is None:
            filtered = dict(pose)
        else:
            filtered = dict(pose)
            for key in keys:
                old_value = float(self.filtered_pose.get(key, pose[key]))
                new_value = float(pose[key])
                filtered[key] = old_value + POSE_FILTER_ALPHA * (new_value - old_value)

        self.filtered_pose = filtered
        self.last_pose = filtered
        return filtered

    def _report_pose(self, now_ms):
        if self.last_pose is None:
            return

        elapsed_ms = (now_ms - self.pose_print_tick) & 0xFFFFFFFF
        if elapsed_ms < POSE_PRINT_INTERVAL_MS:
            return

        pose = self.last_pose
        simple_z = pose.get("simple_z_mm")
        if simple_z is None:
            simple_z_text = "None"
        else:
            simple_z_text = "{:.0f}".format(simple_z)

        print(
            "[pose] x={:.0f}mm y={:.0f}mm z={:.0f}mm "
            "dis={:.0f}mm angle={:.1f}deg simple_z={}mm".format(
                pose["x_mm"],
                pose["y_mm"],
                pose["z_mm"],
                pose["distance_mm"],
                pose["angle_x_deg"],
                simple_z_text,
            )
        )
        self.pose_print_tick = now_ms

    def _draw_result(self, img, points, center, pose=None, measurements=None):
        if points is not None and self.debug_draw_rect:
            for index in range(4):
                x1, y1 = points[index]
                x2, y2 = points[(index + 1) % 4]
                img.draw_line(x1, y1, x2, y2, image.COLOR_GREEN, thickness=2)

        if center is not None and self.debug_draw_err_line:
            img.draw_line(
                CAM_W // 2,
                CAM_H // 2,
                center[0],
                center[1],
                image.COLOR_RED,
                thickness=2,
            )

        if self.debug_draw_err_msg:
            img.draw_string(
                2,
                CAM_H - 32,
                "err: {:4d}, {:4d}".format(
                    self.err_center[0], self.err_center[1]
                ),
                image.COLOR_RED,
                scale=1.5,
                thickness=2,
            )

        # 姿态信息
        line_y = 2
        if pose is not None:
            img.draw_string(
                2, line_y,
                "x:{:.0f} z:{:.0f} a:{:.1f}".format(
                    pose["x_mm"],
                    pose["z_mm"],
                    pose["angle_x_deg"],
                ),
                image.COLOR_GREEN,
                scale=1.3,
                thickness=2,
            )
            line_y += 22

        # 内部图形测量
        if measurements is not None:
            for shape in measurements.get("shapes", []):
                shape_type = shape["type"]
                has_bbox = "bbox" in shape
                if has_bbox:
                    sx, sy, sw, sh = shape["bbox"]

                if shape_type == "circle":
                    # 画圆
                    cx, cy = shape["center"]
                    r = shape["radius_px"]
                    img.draw_circle(cx, cy, r,
                                    image.Color.from_rgb(255, 0, 0), 2)
                    label = "circle"
                    d = shape.get("diameter_mm")
                    if d is not None:
                        label += " D={:.1f}mm".format(d)
                    img.draw_string(cx - 30, cy - r - 18, label,
                                    image.Color.from_rgb(255, 0, 0),
                                    scale=1.2, thickness=2)
                elif shape_type == "tri":
                    # 三角形：画三条边的轮廓，标注边长
                    verts = shape["vertices"]
                    for i in range(3):
                        x1, y1 = verts[i]
                        x2, y2 = verts[(i + 1) % 3]
                        img.draw_line(x1, y1, x2, y2,
                                      image.Color.from_rgb(255, 0, 0), 2)
                    label = "tri"
                    sides = shape.get("sides_mm")
                    if sides:
                        label += " a={:.1f} b={:.1f} c={:.1f}mm".format(
                            sides[0], sides[1], sides[2])
                    # 标签放在三角形最上方顶点之上
                    top_y = min(v[1] for v in verts)
                    top_x = sum(v[0] for v in verts) // 3
                    img.draw_string(top_x - 50, top_y - 18, label,
                                    image.Color.from_rgb(255, 0, 0),
                                    scale=1.2, thickness=2)

                else:
                    # 矩形：画 bounding rect
                    img.draw_rect(sx, sy, sw, sh,
                                  image.Color.from_rgb(255, 0, 0), 2)
                    label = "rect"
                    w = shape.get("width_mm")
                    h = shape.get("height_mm")
                    if w is not None and h is not None:
                        label += " {:.1f}x{:.1f}mm".format(w, h)
                    elif w is not None:
                        label += " w={:.1f}mm".format(w)
                    img.draw_string(sx, sy - 15, label,
                                    image.Color.from_rgb(255, 0, 0),
                                    scale=1.2, thickness=2)

        self.disp.show(img)

    def run(self):
        """返回圆心、画面中心、中心误差、空轮廓列表和本帧有效标志。"""
        self.updated = False
        img = self.cam.read()
        if img is None:
            return [
                self.last_center,
                self.center_pos,
                self.err_center,
                self.last_circle3_points,
                self.updated,
            ]

        frame_bgr = image.image2cv(
            img,
            ensure_bgr=False,
            copy=False,
        )
        process_frame = cv2.resize(
            frame_bgr,
            (PROC_W, PROC_H),
            interpolation=cv2.INTER_AREA,
        )
        process_points = detect_rectangle(process_frame)
        points = scale_points(process_points, PROC_W, PROC_H, CAM_W, CAM_H)
        opencv_valid = points is not None

        self.status_frames += 1
        self.eval_frames += 1
        self.miss_streak, self.yolo_cooldown, should_try_yolo = (
            update_rescue_schedule(
                self.miss_streak,
                self.yolo_cooldown,
                opencv_valid,
                self.yolo_enabled,
            )
        )
        if opencv_valid:
            self.status_opencv_valid += 1
        else:
            self.eval_opencv_invalid += 1
            if should_try_yolo:
                self.status_yolo_attempts += 1
                self.eval_yolo_attempts += 1
                points = self._try_yolo_rescue(img, frame_bgr)
                if points is not None:
                    self.status_yolo_rescues += 1
                    self.eval_yolo_rescues += 1
                    self.miss_streak = 0
                else:
                    self.yolo_cooldown = YOLO_COOLDOWN_FRAMES

        center = (
            diagonal_intersection(points)
            if points is not None
            else None
        )
        raw_pose = estimate_target_pose(points) if points is not None else None
        pose = self._update_pose_filter(raw_pose)

        measurements = (
            measure_internal_shapes(frame_bgr, points, raw_pose)
            if points is not None
            else None
        )

        if center is not None:
            self.last_center = self._to_legacy_center(center)
            self.err_center = self._to_legacy_error(center)
            self.updated = True

        self.last_circle3_points = []
        self._draw_result(img, points, center, pose, measurements)

        now_ms = time.ticks_ms()
        self._update_model_policy(now_ms)
        self._report_status(now_ms)
        self._report_pose(now_ms)
        return [
            self.last_center,
            self.center_pos,
            self.err_center,
            self.last_circle3_points,
            self.updated,
        ]


if __name__ == "__main__":
    disp = display.Display()
    finder = FindRectCircle(disp)
    while not app.need_exit():
        finder.run()
