try:
    import RPi.GPIO as GPIO
except ImportError:
    import fake_rpi
    RPi = fake_rpi.RPi
    GPIO = RPi.GPIO
import time
import threading

class PIRSensorController:

    def __init__(self, pir_pins, states, poll_interval=0.1):
        self.pir_pins = pir_pins
        self.states = states
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()

        GPIO.setmode(GPIO.BCM)
        for pin in pir_pins:
            GPIO.setup(pin, GPIO.IN)

    def start(self):
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop_event.set()
        self.thread.join()
        GPIO.cleanup(self.pir_pins)

    def _poll_loop(self):
        while not self._stop_event.is_set():
            for idx, pin in enumerate(self.pir_pins):
                self.states[idx] = bool(GPIO.input(pin))
            time.sleep(self.poll_interval)
