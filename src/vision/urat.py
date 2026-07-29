# src/vision/urat.py
# UART sender — transmits filtered steel-ball position (mm) to motor controller
import serial


class UARTSender:
    """
    Send steel-ball offset (mm) over UART to the pipe-balancing motor controller.

    Protocol (ASCII, newline-terminated)::

        +012.5\\n    ball 12.5 mm left  of centre
        -003.2\\n    ball  3.2 mm right of centre
        +000.0\\n    ball at centre

    Fixed-width (7 chars + newline) makes parsing trivial on the MCU side.
    """

    def __init__(self, port='/dev/ttyS3', baud=115200, timeout=0.1):
        """
        Args:
            port:    serial device.  Rock4D UART3 → ``/dev/ttyS3``
                     (also try ``/dev/ttyS3`` or ``/dev/ttyAMA3``)
            baud:    baud rate — must match the receiver
            timeout: write timeout (keep short for low latency)
        """
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser = None

    # ------------------------------------------------------------------
    def open(self):
        """Open the serial port.  Returns True on success."""
        try:
            self.ser = serial.Serial(
                self.port, self.baud,
                timeout=self.timeout,
                write_timeout=self.timeout,
            )
            return self.ser.is_open
        except serial.SerialException as e:
            print(f"[UART] failed to open {self.port}: {e}")
            return False

    # ------------------------------------------------------------------
    def close(self):
        """Close the serial port."""
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
            self.ser = None

    # ------------------------------------------------------------------
    def send(self, position_mm):
        """
        Transmit the ball's offset from pipe centre.

        Args:
            position_mm: float — left = positive, right = negative

        Returns:
            True if the write completed without error.
        """
        if self.ser is None or not self.ser.is_open:
            print("[UART] port not open — call open() first")
            return False

        # fixed-width: sign + 3 int digits + dot + 1 frac digit = 7 chars + newline
        # e.g. +012.5  -003.2  +000.0  -125.0
        packet = f"{position_mm:+07.1f}\n".encode('ascii')

        try:
            self.ser.write(packet)
            return True
        except serial.SerialException as e:
            print(f"[UART] write error: {e}")
            return False

    # ------------------------------------------------------------------
    @property
    def is_open(self):
        return self.ser is not None and self.ser.is_open


# ============================================================================
# Standalone test — sends dummy position values
# ============================================================================
if __name__ == "__main__":
    import time
    import math

    PORT = '/dev/ttyS3'  # Rock4D UART3 — adjust if needed

    uart = UARTSender(port=PORT, baud=115200)

    if not uart.open():
        print(f"Cannot open {PORT}.  Check:")
        print("  ls -l /dev/tty*")
        print("  sudo chmod 666 /dev/ttyS3")
        exit(1)

    print(f"[UART] {PORT} open — sending test pattern (Ctrl-C to stop)")

    try:
        t0 = time.time()
        while True:
            elapsed = time.time() - t0
            # simulate a ball oscillating ±120 mm at 0.5 Hz
            fake_mm = 120.0 * math.sin(2 * math.pi * 0.5 * elapsed)
            ok = uart.send(fake_mm)
            if ok:
                print(f"  sent  {fake_mm:+07.1f} mm")
            time.sleep(0.05)  # 20 Hz
    except KeyboardInterrupt:
        print("\n[DONE]")
    finally:
        uart.close()
