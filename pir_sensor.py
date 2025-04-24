try:
    import RPi.GPIO as GPIO
except ImportError:
    # 使用 fake-rpi 提供的 RPi.GPIO 接口
    import fake_rpi
    RPi = fake_rpi.RPi
    GPIO = RPi.GPIO
import time
import threading

class PIRSensorController:
    """
    对应两个 PIR 传感器（GPIO3, GPIO4）的轮询控制器。
    states: 传入一个长度为 2 的 dict {0: bool, 1: bool}，实时更新为当前检测结果。
    """
    def __init__(self, pir_pins, states, poll_interval=0.1):
        self.pir_pins = pir_pins
        self.states = states
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()

        GPIO.setmode(GPIO.BCM)
        for pin in pir_pins:
            GPIO.setup(pin, GPIO.IN)

    def start(self):
        """以守护线程方式启动轮询。"""
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """停止轮询并清理 GPIO。"""
        self._stop_event.set()
        self.thread.join()
        GPIO.cleanup(self.pir_pins)

    def _poll_loop(self):
        while not self._stop_event.is_set():
            for idx, pin in enumerate(self.pir_pins):
                self.states[idx] = bool(GPIO.input(pin))
            time.sleep(self.poll_interval)
