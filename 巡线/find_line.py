"""
视觉巡线 - 多点变权重方案
MaixCAM2 (MaixPy v4)
白底黑线赛道

核心思路：
  在图像纵向划分 N 个水平条带（ROI），每个条带用 find_blobs() 检测黑线中心，
  条带越靠近画面底部（近机器人端）权重越大，加权平均得到平滑的线位置。

算法优势：
  - find_blobs() 是 C 实现，速度极快（相比逐像素 get_pixel() 快 50~100 倍）
  - 多点采样提供"预瞄"能力：远点预判弯道，近点精确纠偏
  - 变权重灵活调节"预瞄 vs 即时响应"的平衡
  - 丢行容错：部分行检测失败不影响整体

@author
@date 2026.7
"""

from maix import camera, display, image, app, time


class LineFollower:
    """多点变权重巡线检测器"""

    # ======================== 可调参数 ========================

    # --- 采样参数 ---
    NUM_STRIPS = 15              # 采样条带数（12~20 推荐）
    STRIP_HEIGHT = 10            # 每条带高度(px)，太小检不出线，太大失去垂直精度
    ROI_TOP = 30                 # 采样区域顶部 Y（跳过远处无用区域）
    ROI_BOTTOM = 230             # 采样区域底部 Y（近机器人端）

    # --- 黑色阈值 ---
    # LAB 色彩空间: L(亮度) < THRESHOLD 视为黑色, AB 不限
    THRESHOLD = 50               # 亮度阈值 (0~100)，越小只识别越黑的像素

    # --- Blob 过滤 ---
    MIN_BLOB_AREA = 8            # 最小 blob 面积(px²)，过滤噪点
    MIN_BLOB_WIDTH = 2           # 最小 blob 宽度(px)，过滤单像素噪点
    MAX_BLOB_WIDTH = 100         # 最大 blob 宽度(px)，过滤大片阴影

    # --- 权重方案 ---
    # 'linear'     : w_i = i+1          -- 默认，近处线性优先
    # 'quadratic'  : w_i = (i+1)^2      -- 近处权重极高，大弯灵敏
    # 'exponential': w_i = 2^i          -- 极度偏重近处
    # 'equal'      : w_i = 1            -- 所有点等权，远距预判为主
    WEIGHT_MODE = 'linear'

    # --- 平滑滤波 ---
    SMOOTH_FACTOR = 0.5           # 低通滤波 (0=不过滤, 越大越平滑但响应越慢)

    # --- find_blobs 加速 ---
    X_STRIDE = 2                  # 水平跳像素采样 (1=不跳, 2=隔一个采一个) 水平隔列采样，计算量减半
    Y_STRIDE = 1                  # 垂直跳像素采样
    DYNAMIC_ROI = True            # 动态ROI：下一帧只在上一帧位置附近搜索（加速+防干扰）
    ROI_MARGIN = 50               # 动态ROI水平窗口半径(px)，黑线左右各留50px

    # --- 调试 ---
    DEBUG = True
    DEBUG_SHOW_STRIPS = False     # 显示条带边界线（影响性能）
    DEBUG_PRINT = False           # 串口打印 error

    # ==========================================================

    def __init__(self, width=320, height=240):
        self._img_w = width
        self._img_h = height
        self._center_x = width / 2.0

        # 预计算条带 Y 坐标和权重表
        self._strip_y = []
        self._weights = []
        self._rebuild_strips()

        # 状态变量
        self.last_error = 0.0
        self.last_center_x = self._center_x
        self.last_points = []          # 上一帧检测点 [(cx, cy, weight), ...]
        self.fps = 0.0
        self._last_strip_cx = {}       # {strip_index: cx} 记录每个条带上一次检测到的线中心

        # 初始化硬件
        self.cam = camera.Camera(width, height, buff_num=1)
        self.disp = display.Display()

    # ==================== 参数更新接口 ====================

    def _rebuild_strips(self):
        """根据当前参数重建条带坐标和权重表"""
        self._strip_y = []
        self._weights = []
        self._last_strip_cx = {}       # 条带位置变了，清空历史

        if self.NUM_STRIPS == 1:
            self._strip_y = [(self.ROI_TOP + self.ROI_BOTTOM) // 2]
        else:
            for i in range(self.NUM_STRIPS):
                y = self.ROI_TOP + i * (self.ROI_BOTTOM - self.ROI_TOP) // (self.NUM_STRIPS - 1)
                self._strip_y.append(int(y))

        for i in range(self.NUM_STRIPS):
            if self.WEIGHT_MODE == 'linear':
                w = i + 1
            elif self.WEIGHT_MODE == 'quadratic':
                w = (i + 1) ** 2
            elif self.WEIGHT_MODE == 'exponential':
                w = 2 ** i
            elif self.WEIGHT_MODE == 'equal':
                w = 1
            else:
                w = i + 1
            self._weights.append(w)

    def set_weight_mode(self, mode):
        """运行时切换权重模式"""
        if mode in ('linear', 'quadratic', 'exponential', 'equal'):
            self.WEIGHT_MODE = mode
            self._rebuild_strips()

    def set_threshold(self, th):
        """运行时调整黑色阈值"""
        self.THRESHOLD = max(0, min(100, th))

    def set_smooth_factor(self, factor):
        """运行时调整平滑系数"""
        self.SMOOTH_FACTOR = max(0.0, min(0.95, factor))

    # ==================== 核心检测 ====================

    def _find_blob_in_strip(self, img, y_center, strip_index):
        """
        在指定条带内寻找黑线 blob。
        支持动态 ROI：如果有上一帧位置记录，先在窄窗口搜索；找不到再退回全宽。
        参数:
            img: maix.image.Image
            y_center: 条带中心 Y 坐标
            strip_index: 条带索引（用于查找上一帧位置）
        返回:
            找到的 blob 对象，或 None
        """
        half_h = self.STRIP_HEIGHT // 2
        y0 = max(0, y_center - half_h)
        y1 = min(img.height(), y_center + half_h)
        h = y1 - y0
        if h <= 0:
            return None

        ths = [[0, self.THRESHOLD, -128, 127, -128, 127]]

        def _search(roi):
            """在给定 ROI 内执行 find_blobs 并返回过滤后的有效列表"""
            try:
                blobs = img.find_blobs(
                    ths, roi=roi,
                    x_stride=self.X_STRIDE, y_stride=self.Y_STRIDE,
                    area_threshold=self.MIN_BLOB_AREA,
                    pixels_threshold=self.MIN_BLOB_AREA,
                    merge=True, margin=3,
                )
            except Exception:
                return []
            if not blobs:
                return []
            # 宽度过滤
            return [b for b in blobs
                    if self.MIN_BLOB_WIDTH <= b.w() <= self.MAX_BLOB_WIDTH]

        # --- 第一遍：动态窄窗口 ---
        if self.DYNAMIC_ROI:
            last_cx = self._last_strip_cx.get(strip_index, None)
            if last_cx is not None:
                x0 = max(0, int(last_cx) - self.ROI_MARGIN)
                x1 = min(img.width(), int(last_cx) + self.ROI_MARGIN)
                narrow_roi = [x0, y0, x1 - x0, h]
                valid = _search(narrow_roi)
                if valid:
                    return self._select_best(valid, strip_index)

        # --- 第二遍：全宽搜索（保底） ---
        full_roi = [0, y0, img.width(), h]
        valid = _search(full_roi)
        if not valid:
            return None
        return self._select_best(valid, strip_index)

    def _select_best(self, valid, strip_index):
        """
        从多个候选 blob 中选择最优的一个：
        - 只有1个 → 直接返回
        - 2个及以上 → 优先选离上一帧位置最近的（防跳变）
        """
        if len(valid) == 1:
            return valid[0]
        last_cx = self._last_strip_cx.get(strip_index, None)
        if last_cx is not None:
            return min(valid, key=lambda b: abs(b.cx() - last_cx))
        return max(valid, key=lambda b: b.area())

    def _detect_all_strips(self, img):
        """
        在所有条带中检测黑线中心
        返回: [(cx, cy, weight, rect_x, rect_y, rect_w, rect_h), ...]
              rect_* 是 blob 在原图中的外接矩形，用于调试框选
        """
        centers = []
        for i, (y, w) in enumerate(zip(self._strip_y, self._weights)):
            blob = self._find_blob_in_strip(img, y, i)
            if blob is not None:
                self._last_strip_cx[i] = blob.cx()  # 记住本帧位置，供下一帧参考
                centers.append((blob.cx(), y, w,
                                blob.x(), blob.y(), blob.w(), blob.h()))
        return centers

    def _weighted_average(self, centers):
        """
        加权平均 + 置信度计算
        centers: [(cx, cy, weight, ...), ...]
        返回: (weighted_x, confidence)
        """
        if not centers:
            return None, 0.0

        total_w = sum(c[2] for c in centers)
        if total_w == 0:
            return None, 0.0

        weighted_x = sum(c[0] * c[2] for c in centers) / total_w
        confidence = len(centers) / self.NUM_STRIPS
        return weighted_x, confidence

    # ==================== 主处理 ====================

    def process(self):
        """
        处理一帧图像
        返回: dict {
            'error': float,       # 线中心相对画面中心的 X 偏移 (px)
            'confidence': float,  # 0.0~1.0, 有效检测点比例
            'center_x': float,    # 加权线中心 X 坐标
            'points': list,       # [(cx, cy, weight), ...] 调试用
            'fps': float,         # 帧率
        }
        """
        # 1. 读摄像头
        img = self.cam.read()

        # 2. 多点检测
        centers = self._detect_all_strips(img)

        # 3. 加权平均
        weighted_x, confidence = self._weighted_average(centers)

        # 4. 误差计算 + 滤波
        if weighted_x is None:
            error = self.last_error
            weighted_x = self.last_center_x
        else:
            raw_error = weighted_x - self._center_x
            # 一阶低通滤波
            error = (1 - self.SMOOTH_FACTOR) * raw_error + self.SMOOTH_FACTOR * self.last_error
            self.last_error = error
            self.last_center_x = weighted_x

        self.last_points = centers

        # 5. FPS（使用 MaixPy 内置 time.fps()，自动跟踪帧率）
        self.fps = time.fps()

        # 6. 调试
        if self.DEBUG:
            self._draw_debug(img, weighted_x, error, confidence)

        self.disp.show(img)

        if self.DEBUG_PRINT:
            print(f"err:{error:+.1f} conf:{confidence:.2f} fps:{self.fps:.0f}")

        return {
            'error': error,
            'confidence': confidence,
            'center_x': weighted_x,
            'points': centers,
            'fps': self.fps,
        }

    # ==================== 调试绘制 ====================

    def _draw_debug(self, img, weighted_x, error, confidence):
        """在图像上叠加调试信息"""
        img_w = img.width()
        img_h = img.height()
        cx_int = int(weighted_x)
        mid = int(self._center_x)

        #画条带边界（可选）
        if self.DEBUG_SHOW_STRIPS:
            for y in self._strip_y:
                half_h = self.STRIP_HEIGHT // 2
                if y - half_h >= 0:
                    img.draw_line(0, y - half_h, img_w, y - half_h,
                                  image.COLOR_GRAY, thickness=1)
                if y + half_h < img_h:
                    img.draw_line(0, y + half_h, img_w, y + half_h,
                                  image.COLOR_GRAY, thickness=1)

        # 画每个检测点：绿色矩形框 + 十字标记
        max_w = max(self._weights) if self._weights else 1
        for pt in self.last_points:
            cx, cy, w = pt[0], pt[1], pt[2]
            rx, ry, rw, rh = pt[3], pt[4], pt[5], pt[6]

            # 画 blob 外接矩形框（绿色，方便观察识别到了什么）
            img.draw_rect(rx, ry, rw, rh, image.COLOR_GREEN, thickness=1)

            # 权重归一化决定十字大小
            size = 3 + int(4 * w / max_w)
            img.draw_cross(int(cx), cy, image.COLOR_GREEN, size=size, thickness=1)

        # 用红色折线连接所有检测点，形成识别到的黑线轨迹
        if len(self.last_points) >= 2:
            for i in range(len(self.last_points) - 1):
                x0, y0 = int(self.last_points[i][0]), self.last_points[i][1]
                x1, y1 = int(self.last_points[i+1][0]), self.last_points[i+1][1]
                img.draw_line(x0, y0, x1, y1, image.COLOR_RED, thickness=2)

        # 画画面中心线（蓝色参考线）
        img.draw_line(mid, self.ROI_TOP, mid, self.ROI_BOTTOM,
                      image.COLOR_BLUE, thickness=1)

        # 画误差向量（从画面中心到线中心，黄色），直观显示偏差大小和方向
        mid_y = (self.ROI_TOP + self.ROI_BOTTOM) // 2
        img.draw_line(mid, mid_y, cx_int, mid_y,
                      image.COLOR_YELLOW, thickness=2)

        # 信息文字（黑底白字增强可读性）
        info_lines = [
            f"err:{error:+.1f}",
            f"conf:{confidence:.2f}",
            f"fps:{self.fps:.0f}",
            f"mode:{self.WEIGHT_MODE}",
        ]
        for i, txt in enumerate(info_lines):
            y = 2 + i * 16
            img.draw_string(3, y + 1, txt, image.COLOR_BLACK, scale=1.0)
            img.draw_string(2, y, txt, image.COLOR_WHITE, scale=1.0)


# ========================= 主函数 =========================

def main():
    """启动巡线检测"""
    follower = LineFollower(width=320, height=240)

    print("=" * 40)
    print("  视觉巡线 - 多点变权重")
    print(f"  分辨率 : {follower._img_w}x{follower._img_h}")
    print(f"  采样条带: {follower.NUM_STRIPS}")
    print(f"  权重模式: {follower.WEIGHT_MODE}")
    print(f"  黑色阈值: {follower.THRESHOLD}")
    print(f"  平滑系数: {follower.SMOOTH_FACTOR}")
    print("=" * 40)
    print("  调试开关:")
    print(f"    DEBUG              = {follower.DEBUG}")
    print(f"    DEBUG_SHOW_STRIPS  = {follower.DEBUG_SHOW_STRIPS}")
    print(f"    DEBUG_PRINT        = {follower.DEBUG_PRINT}")
    print("=" * 40)

    while not app.need_exit():
        result = follower.process()
        # result['error'] 可直接送入 PID 控制器
        # result['confidence'] 可用于判断是否丢失线


if __name__ == "__main__":
    main()
