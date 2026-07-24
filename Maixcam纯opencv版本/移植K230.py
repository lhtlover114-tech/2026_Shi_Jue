"""
MaixCAM2 rectangle target recognition (ported from the K230 version).

Main changes:
- K230 media.sensor/media.display/media.media -> MaixPy camera/display/app.
- Camera frames are requested in BGR888 and exposed to OpenCV with zero copy.
- Detection runs at 320x240 because the original area thresholds were designed
  for that resolution; detected coordinates are scaled back to 640x480.
- The displayed center/error values therefore remain in the 640x480 coordinate
  system, with image center at (320, 240).
"""

import gc
import cv2

from maix import app, camera, display, image, time


CAM_W = 640
CAM_H = 480
CAM_FPS = 60

# The original thresholds are intended for a 320 x 240 processing frame.
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

# Logical RGB colors used by maix.image drawing functions.
COLOR_GRAY = image.Color.from_rgb(160, 160, 160)
COLOR_YELLOW = image.Color.from_rgb(255, 255, 0)
COLOR_RED = image.Color.from_rgb(255, 80, 80)
COLOR_GREEN = image.Color.from_rgb(0, 255, 0)
COLOR_TARGET = image.Color.from_rgb(255, 0, 0)
COLOR_CYAN = image.Color.from_rgb(0, 255, 255)


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

    center = (int(round(center_x)), int(round(center_y)))
    if not (0 <= center[0] < frame_w and 0 <= center[1] < frame_h):
        return None

    return center


def calculate_error(center):
    return center[0] - CAM_W // 2, center[1] - CAM_H // 2


def select_largest_quad(candidates):
    best = None
    best_area = 0

    for points in candidates:
        if not quad_is_target(points):
            continue

        area = polygon_area(points)
        if area > best_area:
            best = points
            best_area = area

    return best


def contour_to_points(approx):
    # OpenCV approxPolyDP normally returns shape (N, 1, 2).
    flat = approx.reshape(-1, 2)
    return [(int(point[0]), int(point[1])) for point in flat]


def select_quad_from_binary(binary):
    contour_result = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Compatible with both OpenCV 3 and OpenCV 4 return signatures.
    contours = contour_result[-2]

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_QUAD_AREA or area > MAX_QUAD_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(contour, APPROX_RATIO * perimeter, True)
        if len(approx) == 4:
            candidates.append(contour_to_points(approx))

    return select_largest_quad(candidates)


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
    _, fixed_binary = cv2.threshold(
        gray, 115, 255, cv2.THRESH_BINARY_INV,
    )
    if border_is_white(fixed_binary):
        fixed_binary = cv2.bitwise_not(fixed_binary)
    result = select_quad_from_binary(fixed_binary)
    if result is not None:
        return result

    # ---- 路径 2: Otsu 全局阈值 ----
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

    # ---- 路径 3: 自适应阈值回退 ----
    adaptive_binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 21, 5,
    )
    if border_is_white(adaptive_binary):
        adaptive_binary = cv2.bitwise_not(adaptive_binary)
    return select_quad_from_binary(adaptive_binary)


def scale_points(points, src_w, src_h, dst_w, dst_h):
    if points is None:
        return None

    scale_x = (dst_w - 1) / max(1, src_w - 1)
    scale_y = (dst_h - 1) / max(1, src_h - 1)

    scaled = []
    for x, y in points:
        dst_x = int(round(x * scale_x))
        dst_y = int(round(y * scale_y))
        dst_x = max(0, min(dst_w - 1, dst_x))
        dst_y = max(0, min(dst_h - 1, dst_y))
        scaled.append((dst_x, dst_y))

    return scaled


def draw_overlay(img, points, center, fps):
    frame_center_x = CAM_W // 2
    frame_center_y = CAM_H // 2

    img.draw_cross(
        frame_center_x,
        frame_center_y,
        COLOR_GRAY,
        size=8,
        thickness=1,
    )
    img.draw_string(
        4,
        4,
        "FPS:%.1f" % fps,
        COLOR_YELLOW,
        scale=1.0,
    )

    if points is None or center is None:
        img.draw_string(
            4,
            24,
            "NO TARGET",
            COLOR_RED,
            scale=1.0,
        )
        return

    for index in range(4):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % 4]
        img.draw_line(x1, y1, x2, y2, COLOR_GREEN, thickness=2)

    center_x, center_y = center
    error_x, error_y = calculate_error(center)

    img.draw_line(
        frame_center_x,
        frame_center_y,
        center_x,
        center_y,
        COLOR_YELLOW,
        thickness=1,
    )
    img.draw_cross(
        center_x,
        center_y,
        COLOR_TARGET,
        size=12,
        thickness=2,
    )
    img.draw_circle(
        center_x,
        center_y,
        4,
        COLOR_TARGET,
        thickness=1,
    )
    img.draw_string(
        4,
        24,
        "Center:(%d,%d)" % (center_x, center_y),
        COLOR_CYAN,
        scale=1.0,
    )
    img.draw_string(
        4,
        44,
        "Error:(%d,%d)" % (error_x, error_y),
        COLOR_YELLOW,
        scale=1.0,
    )


def close_resource(resource, name):
    if resource is None:
        return

    try:
        resource.close()
    except Exception as error:
        print("close %s failed:" % name, error)


def main():
    cam = None
    disp = None
    frame_id = 0
    current_fps = 0.0
    fps_counter = time.FPS(10)

    try:
        # BGR888 avoids an RGB->BGR copy before passing the frame to OpenCV.
        cam = camera.Camera(
            CAM_W,
            CAM_H,
            image.Format.FMT_BGR888,
            fps=CAM_FPS,
            buff_num=3,
        )
        disp = display.Display()

        # Drop unstable frames immediately after camera startup.
        cam.skip_frames(20)

        while not app.need_exit():
            fps_counter.start()

            img = cam.read()
            if img is None:
                continue

            # Zero-copy view. Keep img alive while frame_bgr is in use.
            frame_bgr = image.image2cv(
                img,
                ensure_bgr=False,
                copy=False,
            )

            # Process at 320x240 to match the original thresholds and reduce CPU load.
            process_frame = cv2.resize(
                frame_bgr,
                (PROC_W, PROC_H),
                interpolation=cv2.INTER_AREA,
            )
            process_points = detect_rectangle(process_frame)

            points = scale_points(
                process_points,
                PROC_W,
                PROC_H,
                CAM_W,
                CAM_H,
            )
            center = (
                diagonal_intersection(points)
                if points is not None
                else None
            )
            if center is None:
                points = None

            draw_overlay(img, points, center, current_fps)
            disp.show(img)

            # FPS value is displayed on the next frame.
            current_fps = fps_counter.fps()
            frame_id += 1

            # Release Python/OpenCV references before optional collection.
            process_frame = None
            frame_bgr = None
            img = None

            if frame_id % 120 == 0:
                gc.collect()

    except KeyboardInterrupt:
        print("user stop")
    except Exception as error:
        print("Exception:", error)
    finally:
        close_resource(cam, "camera")
        close_resource(disp, "display")
        gc.collect()


if __name__ == "__main__":
    main()
