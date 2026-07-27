"""Hardware-independent line-following metric helpers."""


def _weighted_x(points):
    """Return weighted X center for (x, y, weight, ...) points, or None."""
    if not points:
        return None

    total_weight = sum(point[2] for point in points)
    if total_weight <= 0:
        return None

    return sum(point[0] * point[2] for point in points) / total_weight


def compute_region_raw_errors(
    centers,
    center_x,
    strip_y,
    region_strip_count,
):
    """
    Compute raw near/far lateral errors from fixed configured strip regions.

    The first ``region_strip_count`` configured Y positions form the far
    region; the last positions form the near region. Missing detections do not
    move the boundaries. Returns ``(near_error, far_error)`` where either value
    is None when that region has no valid weighted point.
    """
    if not strip_y or region_strip_count <= 0:
        return None, None

    count = min(int(region_strip_count), len(strip_y))
    far_max_y = strip_y[count - 1]
    near_min_y = strip_y[-count]

    far_points = [point for point in centers if point[1] <= far_max_y]
    near_points = [point for point in centers if point[1] >= near_min_y]

    near_x = _weighted_x(near_points)
    far_x = _weighted_x(far_points)

    near_error = None if near_x is None else near_x - center_x
    far_error = None if far_x is None else far_x - center_x
    return near_error, far_error
