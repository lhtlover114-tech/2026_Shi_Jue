"""
巡线主程序：视觉检测 + UART 通信 → MSPM0

启动顺序:
  1. LineFollower  初始化摄像头、预计算条带/权重
  2. LineFollowLink 初始化 UART4、启动 TX 线程
  3. 主循环: 视觉检测 → 发布数据 → TX 线程自动发送

使用方法:
  直接运行此文件:
    python main.py

  或导入使用:
    from main import run
    run(debug=True)
"""

from maix import app

from find_line import LineFollower
from uart_link import LineFollowLink


def run(debug=True, width=320, height=240):
    """
    启动巡线系统。

    参数:
        debug:  是否开启屏幕调试显示
        width:  摄像头宽度
        height: 摄像头高度
    """

    # --- 1. 初始化巡线检测 ---
    follower = LineFollower(width=width, height=height)
    follower.DEBUG = debug

    print("=" * 45)
    print("  巡线系统启动")
    print(f"  分辨率   : {follower._img_w}x{follower._img_h}")
    print(f"  采样条带 : {follower.NUM_STRIPS}")
    print(f"  近远条带 : {follower.REGION_STRIPS}")
    print(f"  权重模式 : {follower.WEIGHT_MODE}")
    print(f"  黑色阈值 : {follower.THRESHOLD}")
    print(f"  平滑系数 : {follower.SMOOTH_FACTOR}")
    print(f"  动态 ROI : {'ON' if follower.DYNAMIC_ROI else 'OFF'}")
    print("=" * 45)

    # --- 2. 初始化 UART 通信 ---
    link = LineFollowLink(
        tx_pin="A21",
        rx_pin="A22",
        device="/dev/ttyS4",
        baudrate=460800,
        period_us=10000,         # 100Hz 够用想更快可改为 5000 (200Hz)
        enable_rx=False,         # 暂不需要 MSPM0 回传
    )
    link.DEBUG_PRINT = True   # 开启控制台打印发送数据（调试用，正式跑可关）
    link.start()
    print("[main] UART link started (460800 baud, 100Hz)")

    # --- 3. 接收回调示例（如果 enable_rx=True 时使用） ---
    def on_mspm0_data(frame):
        """处理 MSPM0 发来的数据"""
        # frame 是 32 字节原始帧，可按协议解析
        # 示例：打印第一个命令字节
        if len(frame) >= 3:
            cmd = frame[2]  # 可自定义命令字节位置
            print(f"[main] RX cmd from MSPM0: {cmd:#04x}")

    link.on_rx(on_mspm0_data)

    # --- 4. 主循环 ---
    print("[main] 开始视觉巡线...")

    while not app.need_exit():
        # 视觉检测（一帧的处理时间决定了帧率）
        result = follower.process()
        # result: {'error', 'near_error', 'far_error',
        #          'confidence', 'center_x', 'points', 'fps'}

        # 发布到 MSPM0（瞬时返回，不阻塞视觉）
        if result['confidence'] >= 0.3:
            link.publish_line_data(
                near_error=result['near_error'],
                far_error=result['far_error'],
                confidence=result['confidence'],
                fps=result['fps'],
            )
        else:
            link.publish_line_lost(result['fps'])


if __name__ == "__main__":
    run(debug=True)
