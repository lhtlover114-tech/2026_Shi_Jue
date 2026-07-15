from maix import gpio, time 

# 【控制回路】
# MaixCam (5V)  ------> 继电器 DC+
# MaixCam (GND) ------> 继电器 DC-
# MaixCam (A26) ------> 继电器 IN

# 【动力回路】
# 14V电源正极(+) ------> 继电器 COM
# 继电器 NO      ------> 激光器正极(+)
# 激光器负极(-)  ------> 14V电源负极(-)

class LaserController:
    """
    激光继电器控制模块 (仅输出控制)
    """
    def __init__(self, relay_pin="A26", active_low=False):
        """
        :param relay_pin: 继电器信号连接的引脚 (默认 A26)
        :param active_low: 如果你的继电器是低电平触发，设为 True；高电平触发设为 False
        """
        # 初始化引脚为输出模式
        self.relay = gpio.GPIO(relay_pin, gpio.Mode.OUT)
        self.active_low = active_low
        
        # 初始状态确保关闭
        self.off()

    def on(self):
        """开启激光"""
        val = 0 if self.active_low else 1
        self.relay.value(val)

    def off(self):
        """关闭激光"""
        val = 1 if self.active_low else 0
        self.relay.value(val)

    def set_state(self, state: bool):
        """根据布尔值设置状态：True 为开，False 为关"""
        if state:
            self.on()
        else:
            self.off()

            # ================= 运行测试 =================
def run_laser_test():
    # 1. 初始化 (请根据你的继电器类型修改 active_low)
    # 如果接线后发现 on() 变关，off() 变开，请将 active_low 改为 True
    laser = LaserController(relay_pin="A26", active_low=False)
    
    print("--- 激光控制器测试开始 ---")
    
    try:
        # 场景 1: 基本开关
        print("测试 1: 开启激光 2 秒...")
        laser.on()
        time.sleep(2)
        
        print("测试 1: 关闭激光 1 秒...")
        laser.off()
        time.sleep(1)

        # 场景 2: 循环闪烁 (验证快速响应)
        print("测试 2: 激光闪烁测试 (5次)...")
        for i in range(5):
            print(f"  闪烁 {i+1}")
            laser.on()
            time.sleep(0.5)
            laser.off()
            time.sleep(0.5)

        # 场景 3: 布尔状态控制
        print("测试 3: 使用 set_state 控制...")
        laser.set_state(True)
        time.sleep(1)
        laser.set_state(False)
        
        print("--- 所有测试完成 ---")

    except KeyboardInterrupt:
        print("\n检测到用户停止程序")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 安全退出：无论发生什么，最后都确保激光关闭
        print("正在关闭激光并释放资源...")
        laser.off()

if __name__ == "__main__":
    run_laser_test()