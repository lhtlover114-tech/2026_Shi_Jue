from maix import gpio, pinmap, time

class StepperMotor:
    """MaixPy 步进电机控制类"""

    def __init__(self, step_pin: str, dir_pin: str, en_pin: str, pulse_rate: int = 500, en_active_low: bool = True):
        """
        初始化步进电机
        :param step_pin: STEP 引脚名称 (如 "A14")
        :param dir_pin: DIR 引脚名称 (如 "A18")
        :param en_pin: ENABLE 引脚名称 (如 "A19")
        :param pulse_rate: 速度 (每秒步数/Hz)
        :param en_active_low: 驱动器的使能电平是否为低电平有效 (True=低电平开启电机, False=高电平开启)
        """
        self.pulse_rate = pulse_rate
        self.en_active_low = en_active_low
        
        # 1. 计算延时 (50% 占空比)
        # 周期(us) = 1,000,000 / 频率
        # 半周期(us) = 1,000,000 / 频率 / 2
        self.delay_us = int(1000000 // self.pulse_rate // 2)

        # 2. 设置引脚映射
        # 假设输入 "A14" -> 映射为 "GPIOA14"
        self.step_func = f"GPIO{step_pin}"
        self.dir_func = f"GPIO{dir_pin}"
        self.en_func = f"GPIO{en_pin}"

        try:
            pinmap.set_pin_function(step_pin, self.step_func)
            pinmap.set_pin_function(dir_pin, self.dir_func)
            pinmap.set_pin_function(en_pin, self.en_func)
            print(f"Pinmap set: {step_pin}->{self.step_func}, {dir_pin}->{self.dir_func}, {en_pin}->{self.en_func}")
        except Exception as e:
            print(f"Pinmap Error: {e}")

        # 3. 初始化 GPIO 对象
        self.step_gpio = gpio.GPIO(self.step_func, gpio.Mode.OUT)
        self.dir_gpio = gpio.GPIO(self.dir_func, gpio.Mode.OUT)
        self.en_gpio = gpio.GPIO(self.en_func, gpio.Mode.OUT)

        # 4. 初始化状态
        self.step_gpio.value(0)
        self.disable() # 默认不锁止电机，避免发热
        
    def set_speed(self, rate_hz: int):
        """动态调整速度"""
        self.pulse_rate = rate_hz
        if rate_hz > 0:
            self.delay_us = int(1000000 // rate_hz // 2)

    def enable(self):
        """使能电机（锁止轴）"""
        # 如果是低电平有效(active_low=True)，则输出0表示启用
        val = 0 if self.en_active_low else 1
        self.en_gpio.value(val)
        # print("Motor Enabled")

    def disable(self):
        """释放电机（脱力）"""
        # 如果是低电平有效(active_low=True)，则输出1表示禁用
        val = 1 if self.en_active_low else 0
        self.en_gpio.value(val)
        # print("Motor Disabled")

    def drive(self, steps: int, direction: bool):
        """
        驱动电机转动
        :param steps: 脉冲数量
        :param direction: True/False 方向
        """
        # 1. 确保电机已使能
        self.enable()
        
        # 2. 设置方向
        self.dir_gpio.value(1 if direction else 0)
        
        # 3. 发送脉冲
        # 使用本地变量减少 self 查找开销，提高 Python 循环速度
        step_pin = self.step_gpio
        delay = self.delay_us
        
        for _ in range(steps):
            step_pin.value(1)
            time.sleep_us(delay)
            step_pin.value(0)
            time.sleep_us(delay)

        # 运动完成后，通常保持 Enable 状态以维持位置力矩
        # 如果需要省电，可以在这里调用 self.disable()



# # ==========================================
# # 主程序调用逻辑
# # ==========================================

# # 硬件接线配置
# PIN_STEP = "A14"
# PIN_DIR  = "A18"
# PIN_EN   = "A19"

# # 驱动器配置：大多数驱动器(如A4988)是低电平有效(EN接GND或输出0时工作)
# ENABLE_ACTIVE_LOW = True 
# SPEED_HZ = 800  # 速度：每秒800步 (根据细分设置，如果是16细分，这大约是1/4转每秒)

# print("Initializing Stepper Motor...")
# motor = StepperMotor(step_pin=PIN_STEP, 
#                      dir_pin=PIN_DIR, 
#                      en_pin=PIN_EN, 
#                      pulse_rate=SPEED_HZ, 
#                      en_active_low=ENABLE_ACTIVE_LOW)

# try:
#     while True:
#         print("Rotate Direction A (1600 steps)")
#         # 转动 1600 步，方向 True
#         motor.drive(3200, True)
        
#         time.sleep(0.5) # 停顿 0.5 秒
        
#         print("Rotate Direction B (1600 steps)")
#         # 反向转动 1600 步，方向 False
#         motor.drive(3200, False)
        
#         time.sleep(0.5)

# except KeyboardInterrupt:
#     print("Stopped by user")
#     motor.disable() # 退出时释放电机