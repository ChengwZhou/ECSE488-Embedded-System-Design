import RPi.GPIO as GPIO
import time

# 把摄像头索引映射到对应的 PIR GPIO 引脚
PIR_PINS = {
    0: 17,
    1: 27,
}

GPIO.setmode(GPIO.BCM)
for pin in PIR_PINS.values():
    GPIO.setup(pin, GPIO.IN)

def is_available(cam_idx):
    """PIR 只要定义了对应引脚就认为可用"""
    return cam_idx in PIR_PINS

def get_presence(cam_idx):
    """
    直接读 GPIO 高/低电平：
    高电平说明该区域有人或物体在移动。
    """
    pin = PIR_PINS.get(cam_idx)
    if pin is None:
        return False
    return GPIO.input(pin) == GPIO.HIGH

if __name__ == '__main__':
    # 简单自测
    print("按 Ctrl+C 退出")
    try:
        while True:
            for idx in PIR_PINS:
                print(f"Camera{idx} presence:", get_presence(idx))
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()
