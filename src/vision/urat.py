# src/vision/urat.py
# High-speed threaded UART sender — 1 kHz updates for motor controller
import serial
import threading
import time


class UARTSender:
    """
    Threaded UART sender for steel-ball position (mm).

    - Main thread calls ``update(position_mm)`` at any rate (e.g. 120 Hz).
    - A dedicated sender thread transmits the *latest* value at a fixed
      rate (default 1000 Hz) so the motor controller always has fresh data,
      even between camera frames.

    Protocol (ASCII, fixed-width, newline-terminated)::

        +012.5\\n    ball 12.5 mm left  of centre
        -003.2\\n    ball  3.2 mm right of centre
    """

    PACKET_FMT = "{:+07.1f}\n"   # 7 chars + newline = 8 bytes

    def __init__(self, port='/dev/ttyS3', baud=115200):
        """
        Args:
            port: serial device (e.g. /dev/ttyS3, /dev/ttyACM0)
            baud: baud rate — 115200 @ 8-byte packets supports ~1400 Hz
        """
        self.port = port
        self.baud = baud
        self._serial = None
        self._value = 0.0
        self._has_update = False
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._tx_errors = 0
        self._tx_count = 0

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def open(self):
        """Open serial port. Returns True on success."""
        try:
            self._serial = serial.Serial(
                self.port, self.baud,
                timeout=0,
                write_timeout=0,
            )
            return True
        except serial.SerialException as e:
            print(f"[UART] failed to open {self.port}: {e}")
            return False

    def start(self, rate_hz=1000):
        """Launch the sender thread at *rate_hz* updates per second."""
        if not self._serial or not self._serial.is_open:
            print("[UART] cannot start — port not open")
            return False
        self._running = True
        self._thread = threading.Thread(target=self._run, args=(rate_hz,),
                                        daemon=True, name="uart-tx")
        self._thread.start()
        return True

    def stop(self):
        """Stop the sender thread and close the port."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
            self._serial = None

    # ------------------------------------------------------------------
    # main-thread API
    # ------------------------------------------------------------------

    def update(self, position_mm):
        """
        Post the latest ball position.  Call this from the main loop
        at whatever rate you have new data (typically 30–120 Hz).
        """
        with self._lock:
            self._value = position_mm
            self._has_update = True

    # ------------------------------------------------------------------
    # sender thread
    # ------------------------------------------------------------------

    def _run(self, rate_hz):
        interval = 1.0 / rate_hz
        next_tick = time.perf_counter()

        while self._running:
            # grab the latest value
            with self._lock:
                val = self._value

            packet = self.PACKET_FMT.format(val).encode('ascii')

            try:
                self._serial.write(packet)
                self._tx_count += 1
            except serial.SerialException:
                self._tx_errors += 1

            # ---- precise timing: sleep + spin-wait ----
            next_tick += interval
            now = time.perf_counter()
            remaining = next_tick - now

            if remaining > 0.0005:
                # sleep for all but the last 200 µs
                time.sleep(remaining - 0.0002)
            if remaining > 0:
                # spin-wait the final slice for µs precision
                while time.perf_counter() < next_tick:
                    pass
            else:
                # missed the tick — reset to next interval, no burst
                next_tick = time.perf_counter() + interval

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------

    @property
    def tx_errors(self):
        return self._tx_errors

    @property
    def tx_count(self):
        return self._tx_count

    @property
    def is_open(self):
        return self._serial is not None and self._serial.is_open


# ============================================================================
# Standalone test
# ============================================================================
if __name__ == "__main__":
    import math

    PORT = '/dev/ttyS3'

    uart = UARTSender(port=PORT, baud=115200)
    if not uart.open():
        print(f"Cannot open {PORT}")
        exit(1)

    uart.start(rate_hz=1000)
    print(f"[UART] {PORT} — sending at 1000 Hz (Ctrl-C to stop)")

    try:
        t0 = time.perf_counter()
        while True:
            elapsed = time.perf_counter() - t0
            fake_mm = 120.0 * math.sin(2 * math.pi * 0.5 * elapsed)
            uart.update(fake_mm)
            time.sleep(0.008)  # simulate ~120 Hz camera
    except KeyboardInterrupt:
        pass
    finally:
        uart.stop()
        print(f"\n[DONE]  tx errors: {uart.tx_errors}")
