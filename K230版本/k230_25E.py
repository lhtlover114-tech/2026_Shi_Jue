import gc
import os
import time
import cv2

from media.sensor import *
from media.display import *
from media.media import *


IMG_W = 640
IMG_H = 480

# Target filters for a 320 x 240 processing frame.
MIN_QUAD_AREA = 3500
MAX_QUAD_AREA = 70000
EDGE_MARGIN = 4
MIN_EDGE_LENGTH_SQ = 30 * 30
MAX_CORNER_COS_SQ = 60
MAX_OPPOSITE_EDGE_RATIO = 4
MIN_BBOX_FILL_PERCENT = 50
APPROX_RATIO = 0.025


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


def quad_is_target(points):
    if len(points) != 4 or not is_convex_quad(points):
        return False

    x, y, width, height = bbox_from_points(points)
    if width <= 0 or height <= 0:
        return False
    if x <= EDGE_MARGIN or y <= EDGE_MARGIN:
        return False
    if x + width >= IMG_W - EDGE_MARGIN or y + height >= IMG_H - EDGE_MARGIN:
        return False

    area = polygon_area(points)
    if area < MIN_QUAD_AREA or area > MAX_QUAD_AREA:
        return False
    if area * 100 < width * height * MIN_BBOX_FILL_PERCENT:
        return False
    return has_rectangle_geometry(points)


def diagonal_intersection(points):
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
    if not (0 <= center[0] < IMG_W and 0 <= center[1] < IMG_H):
        return None
    return center


def calculate_error(center):
    return center[0] - IMG_W // 2, center[1] - IMG_H // 2


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
    try:
        approx[0][0][0]
        nested_points = True
    except TypeError:
        nested_points = False

    if nested_points:
        return [(int(point[0][0]), int(point[0][1])) for point in approx]
    return [(int(point[0]), int(point[1])) for point in approx]


def select_quad_from_binary(binary):
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_QUAD_AREA or area > MAX_QUAD_AREA:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, APPROX_RATIO * perimeter, True)
        if len(approx) == 4:
            candidates.append(contour_to_points(approx))

    return select_largest_quad(candidates)


def border_is_white(binary):
    margin = 2
    samples = (
        (margin, margin),
        (IMG_W - margin - 1, margin),
        (margin, IMG_H - margin - 1),
        (IMG_W - margin - 1, IMG_H - margin - 1),
    )
    white_count = 0
    for x, y in samples:
        if int(binary[y][x]) != 0:
            white_count += 1
    return white_count >= 3


def detect_rectangle(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    otsu_level, normal_binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    use_inverse = border_is_white(normal_binary)
    if use_inverse:
        _, primary_binary = cv2.threshold(
            gray, otsu_level, 255, cv2.THRESH_BINARY_INV
        )
    else:
        primary_binary = normal_binary

    return select_quad_from_binary(primary_binary)


def draw_overlay(img, points, center, fps):
    frame_center_x = IMG_W // 2
    frame_center_y = IMG_H // 2

    img.draw_cross(
        frame_center_x,
        frame_center_y,
        color=(160, 160, 160),
        size=8,
        thickness=1,
    )
    img.draw_string_advanced(
        4, 4, 16, "FPS:%.1f" % fps, color=(255, 255, 0)
    )

    if points is None or center is None:
        img.draw_string_advanced(
            4, 24, 16, "NO TARGET", color=(255, 80, 80)
        )
        return

    for index in range(4):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % 4]
        img.draw_line(
            x1, y1, x2, y2, color=(0, 255, 0), thickness=2
        )

    center_x, center_y = center
    error_x, error_y = calculate_error(center)
    img.draw_line(
        frame_center_x,
        frame_center_y,
        center_x,
        center_y,
        color=(255, 255, 0),
        thickness=1,
    )
    img.draw_cross(
        center_x,
        center_y,
        color=(255, 0, 0),
        size=12,
        thickness=2,
    )
    img.draw_circle(
        center_x,
        center_y,
        4,
        color=(255, 0, 0),
        thickness=1,
    )
    img.draw_string_advanced(
        4,
        24,
        16,
        "Center:(%d,%d)" % (center_x, center_y),
        color=(0, 255, 255),
    )
    img.draw_string_advanced(
        4,
        44,
        16,
        "Error:(%d,%d)" % (error_x, error_y),
        color=(255, 255, 0),
    )


def cleanup_resources(sensor, display_attempted, media_attempted):
    errors = []

    if sensor is not None:
        try:
            sensor.stop()
        except BaseException as error:
            errors.append(("sensor.stop", error))

    if display_attempted:
        try:
            Display.deinit()
        except BaseException as error:
            errors.append(("Display.deinit", error))

    try:
        os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
    except BaseException as error:
        errors.append(("os.exitpoint", error))

    try:
        time.sleep_ms(100)
    except BaseException as error:
        errors.append(("time.sleep_ms", error))

    if media_attempted:
        try:
            MediaManager.deinit()
        except BaseException as error:
            errors.append(("MediaManager.deinit", error))

    return errors


def main():
    sensor = None
    frame_id = 0
    clock = time.clock()
    display_attempted = False
    media_attempted = False

    try:
        os.exitpoint(os.EXITPOINT_ENABLE)

        sensor = Sensor(width=IMG_W, height=IMG_H, fps=60)
        sensor.reset()
        sensor.set_framesize(width=IMG_W, height=IMG_H)
        sensor.set_pixformat(Sensor.RGB888)

        display_attempted = True
        Display.init(
            Display.VIRT,
            width=IMG_W,
            height=IMG_H,
            fps=60,
            to_ide=True,
        )
        media_attempted = True
        MediaManager.init()
        sensor.run()

        while True:
            os.exitpoint()
            clock.tick()
            frame_id += 1

            img = sensor.snapshot()
            points = detect_rectangle(img.to_numpy_ref())
            center = diagonal_intersection(points) if points is not None else None
            if center is None:
                points = None

            draw_overlay(img, points, center, clock.fps())
            Display.show_image(img)
            img = None

            if frame_id % 120 == 0:
                gc.collect()

    except KeyboardInterrupt:
        print("user stop")
    except BaseException as error:
        print("Exception:", error)
    finally:
        for resource, error in cleanup_resources(
            sensor, display_attempted, media_attempted
        ):
            print("Cleanup error:", resource, error)


if __name__ == "__main__":
    main()
