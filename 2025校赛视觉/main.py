from maix import image, camera, display, app, time, gpio
import sys

# ================= 1. 导入驱动库 =================
# 导入刚才保存的步进电机驱动
try:
    from stepper_driver import StepperMotor
except ImportError:
    print("请确保 stepper_driver.py 文件已上传")
    sys.exit(0)

# 导入激光模块
try:
    from laser import LaserController
except ImportError:
    print("请确保 laser.py 文件已上传")
    # 如果没有laser.py，防止报错定义一个空类
    class LaserController:
        def __init__(self, **kwargs): pass
        def on(self): pass
        def off(self): pass

from pid_util import PID
from find_circle import FindRectCircle

# ================= 2. 初始化显示与界面 =================
# 初始化显示
disp = display.Display()
img = image.Image(disp.width(), disp.height())
msg = "Auto-Aiming System"
# 使用白色文字
size = image.string_size(msg, scale=1.5, thickness=2)
img.draw_string((img.width() - size.width()) // 2, 
                (img.height() - size.height()) // 2, 
                msg, scale=1.5, color=image.Color.from_rgb(255, 255, 255), thickness=2)
disp.show(img)

SHOW_DEBUG = True

# ================= 3. 类定义 (滤波、云台、视觉) =================

class SimpleKalmanFilter:
    """
    简化的一维卡尔曼滤波器
    """
    def __init__(self, R, Q, initial_value=0.0):
        """
        :param R: 测量噪声协方差 (Measurement Noise) - 越大代表越不信任传感器(画面)
        :param Q: 过程噪声协方差 (Process Noise) - 越大代表目标运动越剧烈
        """
        self.R = R 
        self.Q = Q
        self.P = 1.0       # 估计协方差 (初始不确定性)
        self.x = initial_value # 状态估计值
        self.K = 0.0       # 卡尔曼增益

    def update(self, measurement):
        # 1. 预测 (Prediction)
        # 假设下一刻的位置和这一刻一样 (对于云台追踪，这是一个合理的短时假设)
        # self.x = self.x 
        self.P = self.P + self.Q

        # 2. 更新 (Update)
        # 计算卡尔曼增益
        self.K = self.P / (self.P + self.R)
        # 修正估计值
        self.x = self.x + self.K * (measurement - self.x)
        # 更新协方差
        self.P = (1 - self.K) * self.P

        return self.x
    
    # [新增] 重置滤波器状态（用于从扫描切换到追踪时）
    def reset(self, value=0.0):
        self.x = value
        self.P = 1.0

class Gimbal:
    def __init__(self, pitch: StepperMotor, pid_pitch: PID,
                 yaw: StepperMotor, pid_yaw: PID, max_steps: int):
        self._pitch = pitch
        self._yaw = yaw
        self._pid_pitch = pid_pitch
        self._pid_yaw = pid_yaw
        self.max_steps = max_steps # 动态限制最大步数

    def run(self, pitch_err: float, yaw_err: float,
            pitch_reverse: bool = False, yaw_reverse: bool = False):
        
        # --- Pitch (俯仰) ---
        out_pitch = self._pid_pitch.get_pid(pitch_err, 1)
        steps_pitch = int(abs(out_pitch))
        
        # 限制最大步数（防止图像卡顿）
        if steps_pitch > self.max_steps:
            steps_pitch = self.max_steps
            
        if steps_pitch > 0:
            dir_pitch = (out_pitch > 0)
            if pitch_reverse:
                dir_pitch = not dir_pitch
            self._pitch.drive(steps_pitch, dir_pitch)

        # [恢复] Pitch 轴调试打印
        if SHOW_DEBUG and steps_pitch > 0:
            print(f"P_Err:{int(pitch_err)}->Stps:{steps_pitch}", end=", ")

        # --- Yaw (偏航) ---
        out_yaw = self._pid_yaw.get_pid(yaw_err, 1)
        steps_yaw = int(abs(out_yaw))
        
        if steps_yaw > self.max_steps:
            steps_yaw = self.max_steps

        if steps_yaw > 0:
            dir_yaw = (out_yaw > 0)
            if yaw_reverse:
                dir_yaw = not dir_yaw
            self._yaw.drive(steps_yaw, dir_yaw)

        # [恢复] Yaw 轴调试打印
        if SHOW_DEBUG and steps_yaw > 0:
            print(f"Y_Err:{int(yaw_err)}->Stps:{steps_yaw}")

    # [扫描功能] 仅旋转 Yaw 轴
    def scan_yaw(self, steps, direction):
        self._yaw.drive(steps, direction)


class Target:
    """
    目标识别与误差计算类
    """
    def __init__(self, out_range_pitch: float, out_range_yaw: float, ignore_limit: float, disp: display.Display):
        self.pitch = 0
        self.yaw = 0
        self.out_range_pitch = out_range_pitch
        self.out_range_yaw = out_range_yaw
        self.ignore = ignore_limit

        self.finder = FindRectCircle(disp)
        self.w, self.h = self.finder.get_res()

    def _get_target_err_pixels(self):
            # 获取 find_circle.py 的所有返回值
            # results = [last_center, center_pos, err_center, circle3, updated]
            results = self.finder.run()
            
            if results and len(results) >= 5:
                err_center = results[2]
                is_valid_update = results[4] # 获取 updated 标志位
                
                # 只有当算法认为“此次成功更新了圆心”时，才返回 True
                if is_valid_update:
                    return err_center, True 
                    
            return (0, 0), False

    def get_target_err(self):
        # [修改] 解包返回值
        (err_x, err_y), is_found = self._get_target_err_pixels()
        
        if not is_found:
            return 0, 0, False # 明确告诉主程序没找到

        # 将像素误差映射到 [-out_range, +out_range]
        # 注意：这里 Y 轴通常是反的，或者需要在电机反转里处理，这里暂且按标准映射
        self.pitch = (err_y / self.h) * self.out_range_pitch
        self.yaw = (err_x / self.w) * self.out_range_yaw
        
        # 死区处理
        if abs(self.pitch) < self.out_range_pitch * self.ignore:
            self.pitch = 0
        if abs(self.yaw) < self.out_range_yaw * self.ignore:
            self.yaw = 0
            
        return self.pitch, self.yaw, True # 找到了


# ================= 4. 主程序 =================
def main(disp):
    # ================= 1. 物理参数配置 =================
    
    # 电机参数
    STEP_ANGLE = 1.8        # 电机步距角
    MICROSTEPS = 16         # 驱动器细分设置
    STEPS_PER_REV = int(360 / STEP_ANGLE * MICROSTEPS) # 一圈的总步数 = 3200
    STEPS_PER_DEGREE = STEPS_PER_REV / 360.0           # 1度对应的步数 ≈ 8.88步
    
    # 速度与性能平衡
    MOTOR_SPEED_HZ = 3200   # 设为 3200Hz 意味着 1秒转1圈，速度较快
    
    # 计算每次循环允许的最大步数： 
    MAX_STEPS_LIMIT = int(0.03 * MOTOR_SPEED_HZ) 
    print(f"Physics: 1 Degree = {STEPS_PER_DEGREE:.2f} Steps")
    print(f"Logic: Max Steps/Loop = {MAX_STEPS_LIMIT} (approx {MAX_STEPS_LIMIT/STEPS_PER_DEGREE:.1f} degrees)")

    # 摄像头参数估算 (用于计算PID的P值)
    CAMERA_FOV_DEGREE = 80.0 
    
    # ================= 2. 引脚配置 =================
    PITCH_STEP_PIN, PITCH_DIR_PIN, PITCH_EN_PIN = "A14", "A18", "A19"
    YAW_STEP_PIN, YAW_DIR_PIN, YAW_EN_PIN       = "A15", "A16", "A17"

    # [激光控制]
    # 请根据继电器实际电平修改 active_low (True=低电平开, False=高电平开)
    laser = LaserController(relay_pin="A26", active_low=False)

    # ================= 3. PID 参数自动计算 =================
    target_err_range = 100.0  # 我们将画面误差归一化为 -100 到 100
    
    half_fov_steps = (CAMERA_FOV_DEGREE / 2) * STEPS_PER_DEGREE
    suggested_p = half_fov_steps / target_err_range
    P_VAL = suggested_p * 12
    
    print(f"Auto-Tuning PID P-gain: {P_VAL:.2f}")

    # [P, I, D, I_max]P原始42
    pitch_pid_params = [P_VAL*0.5, 0, P_VAL * 0.15, 100] 
    yaw_pid_params   = [P_VAL*0.5, 0, P_VAL * 0.15, 100]

    target_ignore_limit = 0.03 # 死区 3%

    pitch_reverse = True  # 如果发现上下反了，改为 True
    yaw_reverse = True   # 如果发现左右反了，改为 True

    # ================= 4. 卡尔曼滤波参数  =================
    KF_R = 1   # 假设相机噪声适中
    KF_Q = 0.1   # 假设目标运动是平滑的
    
    kf_pitch = SimpleKalmanFilter(R=KF_R, Q=KF_Q)
    kf_yaw   = SimpleKalmanFilter(R=KF_R, Q=KF_Q)

    # ================= 初始化 =================
    print("Init Motors...")
    pitch_motor = StepperMotor(PITCH_STEP_PIN, PITCH_DIR_PIN, PITCH_EN_PIN, 
                               pulse_rate=MOTOR_SPEED_HZ, en_active_low=True)
    yaw_motor = StepperMotor(YAW_STEP_PIN, YAW_DIR_PIN, YAW_EN_PIN, 
                             pulse_rate=MOTOR_SPEED_HZ, en_active_low=True)
    
    target = Target(target_err_range, target_err_range, target_ignore_limit, disp)
    pid_pitch = PID(*pitch_pid_params)
    pid_yaw = PID(*yaw_pid_params)
    gimbal = Gimbal(pitch_motor, pid_pitch, yaw_motor, pid_yaw, MAX_STEPS_LIMIT)

    print("System Ready. Mode: Auto (Scan -> Track)")
    pitch_motor.enable()
    yaw_motor.enable()
    
    # 确保启动时激光是关的
    laser.off() 

    # ================= 5. 扫描逻辑变量 =================
    SCAN_STEPS = 100        # 每次循环转动的步数 (扫描速度)
    SCAN_DIR = True        # 扫描方向 (True正转, False反转)
    SCAN_MAX_REV = 5     # 最大扫描圈数 (防止绕线)
    scan_accumulated_steps = 0   
    max_scan_steps = int(STEPS_PER_REV * SCAN_MAX_REV) 

    try:
        while not app.need_exit():
            # 1. 获取目标 (核心：这里决定了是追踪还是扫描)
            # 返回: pitch误差, yaw误差, 是否找到目标
            raw_err_pitch, raw_err_yaw, is_found = target.get_target_err()

            if is_found:
                # ==========================================
                #      状态 A: 找到目标 (Target Found)
                # ==========================================
                
                # 1. 【核心需求】找到目标后，立即开启激光
                laser.on()
                
                # 2. 数据滤波
                filtered_pitch = kf_pitch.update(raw_err_pitch)
                filtered_yaw   = kf_yaw.update(raw_err_yaw)

                # 3. PID 闭环控制 (追踪打击)
                gimbal.run(filtered_pitch, filtered_yaw, 
                           pitch_reverse=pitch_reverse, 
                           yaw_reverse=yaw_reverse)
                
                # 4. 既然找到了，重置扫描计数器
                scan_accumulated_steps = 0 

            else:
                # ==========================================
                #      状态 B: 没找到目标 (Scanning)
                # ==========================================
                
                # 1. 【核心需求】没找到目标，强制关闭激光
                laser.off()
                
                # 2. 重置滤波器
                kf_pitch.reset(0)
                kf_yaw.reset(0)

                # 3. 执行扫描逻辑 (仅 Yaw 轴水平旋转)
                if scan_accumulated_steps < max_scan_steps:
                    gimbal.scan_yaw(SCAN_STEPS, SCAN_DIR)
                    scan_accumulated_steps += SCAN_STEPS
                else:
                    # 如果转了一圈多还没找到，反向再扫
                    SCAN_DIR = not SCAN_DIR
                    scan_accumulated_steps = 0
                    print("Scan direction flipped.")

    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        print("Stopping System...")
        laser.off()          # 退出时务必关激光
        pitch_motor.disable()
        yaw_motor.disable()

if __name__ == '__main__':
    main(disp)