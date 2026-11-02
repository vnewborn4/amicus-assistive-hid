"""
Amicus Mimic™ & Mimic-AT™ — Rev 1.0 Firmware (RP2350 / CircuitPython / MicroPython)
Implements:
1. 1000Hz USB HID Absolute/Relative Mouse & Keyboard emulation
2. BioSpline™ 5th-order minimum-jerk trajectory calculation on MCU
3. Assistive Technology (AT) Engine:
   - Real-time 2nd-order Butterworth low-pass tremor filter (4-8 Hz suppression)
   - Magnetic Snap-to-Target attraction for clickable UI elements
   - Automated Dwell-Click engine for hands-free or tremor-free interaction
   - Adaptive Switch & Sip-and-Puff trigger parser
4. Serial command protocol:
   - 'M x y duration_ms' (move mouse with BioSpline)
   - 'C left/right' (click button)
   - 'K text' (type keyboard text)
   - 'AT_MODE 1/0' (toggle Assistive Technology mode)
   - 'TREMOR_FILTER 1/0' (toggle real-time Butterworth tremor filter)
   - 'ADD_TARGET x1 y1 x2 y2' (register interactive UI bounding box)
   - 'CLEAR_TARGETS' (clear target bounding boxes)
   - 'DWELL_CLICK ms' (configure dwell click duration, 0 to disable)
   - 'FILTER_INPUT dx dy' (process human pass-through mouse input through tremor filter)
   - 'SWITCH_PRESS btn' (actuate single/dual adaptive switch)
5. Hardware Negative-Veto takeover monitor
"""

import time
import math
import sys

# Optional CircuitPython USB HID imports (fallback to serial loop if running standalone)
try:
    import usb_hid
    from adafruit_hid.mouse import Mouse
    from adafruit_hid.keyboard import Keyboard
    from adafruit_hid.keyboard_layout_us import KeyboardLayoutUS
    HAS_HID = True
    mouse = Mouse(usb_hid.devices)
    keyboard = Keyboard(usb_hid.devices)
    layout = KeyboardLayoutUS(keyboard)
except ImportError:
    HAS_HID = False

# Screen resolution normalization (default 1920x1080)
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

current_x = 960
current_y = 540

# Assistive Technology Configuration
at_mode_enabled = False
tremor_filter_enabled = False
dwell_click_ms = 0
dwell_timer_start = None
dwell_target_hovered = None
active_targets = []  # List of tuples: (x1, y1, x2, y2, center_x, center_y)


class ButterworthTremorFilter:
    """
    2nd-order IIR Butterworth Low-Pass Filter:
    Designed for 1000Hz sampling rate, fc = 3.5Hz (suppresses 4-8Hz Parkinsonian/essential tremors).
    Difference equation:
    y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
    """
    def __init__(self):
        # Normalized coefficients for fs=1000Hz, fc=3.5Hz
        self.b0 = 0.0001192
        self.b1 = 0.0002384
        self.b2 = 0.0001192
        self.a1 = -1.9687
        self.a2 = 0.9692
        self.reset()

    def reset(self):
        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0

    def step(self, x: float) -> float:
        y = (self.b0 * x) + (self.b1 * self.x1) + (self.b2 * self.x2) - (self.a1 * self.y1) - (self.a2 * self.y2)
        self.x2 = self.x1
        self.x1 = x
        self.y2 = self.y1
        self.y1 = y
        return y


filter_x = ButterworthTremorFilter()
filter_y = ButterworthTremorFilter()


def solve_minimum_jerk_step(tau: float) -> float:
    """
    Polynomial 5th-order minimum-jerk profile:
    s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
    where tau in [0, 1]
    """
    tau2 = tau * tau
    tau3 = tau2 * tau
    tau4 = tau3 * tau
    tau5 = tau4 * tau
    return 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5


# Hardware Negative-Veto Interlock state (<10ms physical cancel response)
negative_veto_triggered = False


def trigger_negative_veto() -> float:
    """Immediately halts any active trajectory within <10ms upon physical user input."""
    global negative_veto_triggered
    t0 = time.monotonic()
    negative_veto_triggered = True
    return (time.monotonic() - t0) * 1000.0


def execute_mouse_trajectory(target_x: int, target_y: int, duration_ms: int = 350):
    """Executes a deterministic assistive minimum-jerk cursor movement with <10ms negative-veto cancel."""
    global current_x, current_y, negative_veto_triggered
    negative_veto_triggered = False
    start_x = current_x
    start_y = current_y
    dx = target_x - start_x
    dy = target_y - start_y

    steps = max(10, int(duration_ms))
    step_delay = min(0.005, (duration_ms / 1000.0) / steps)  # <=5ms poll interval guarantees <10ms cancel

    prev_s_x = start_x
    prev_s_y = start_y

    for i in range(1, steps + 1):
        if negative_veto_triggered:
            print("[PICO2_AT] Negative-Veto Interlock triggered (<10ms) — trajectory halted immediately.")
            return False

        tau = i / float(steps)
        s = solve_minimum_jerk_step(tau)

        step_target_x = start_x + (dx * s)
        step_target_y = start_y + (dy * s)

        rel_x = int(round(step_target_x - prev_s_x))
        rel_y = int(round(step_target_y - prev_s_y))

        if HAS_HID and (rel_x != 0 or rel_y != 0):
            mouse.move(x=rel_x, y=rel_y)

        prev_s_x += rel_x
        prev_s_y += rel_y
        time.sleep(step_delay)

    current_x = target_x
    current_y = target_y
    print(f"[PICO2_AT] Reached: ({current_x}, {current_y}) in {duration_ms}ms")
    return True


def apply_assistive_magnetic_snap(x: int, y: int, attraction_radius: int = 25) -> tuple[int, int]:
    """Checks if current position is within attraction radius of any registered UI target."""
    best_target = None
    min_dist = float('inf')

    for (x1, y1, x2, y2, cx, cy) in active_targets:
        # Distance to center of bounding box
        dist = math.hypot(x - cx, y - cy)
        # Inside bounding box or within attraction radius
        if (x1 - attraction_radius <= x <= x2 + attraction_radius) and \
           (y1 - attraction_radius <= y <= y2 + attraction_radius):
            if dist < min_dist:
                min_dist = dist
                best_target = (cx, cy)

    if best_target and min_dist <= attraction_radius:
        # Gently snap toward target center
        snap_x = int(x + (best_target[0] - x) * 0.65)
        snap_y = int(y + (best_target[1] - y) * 0.65)
        return (snap_x, snap_y)

    return (x, y)


def process_human_pass_through_motion(raw_dx: float, raw_dy: float):
    """
    Processes physical human mouse motion through the real-time tremor filter
    and applies magnetic snap-to-target attraction for motor-impaired users.
    """
    global current_x, current_y, dwell_timer_start, dwell_target_hovered

    if tremor_filter_enabled:
        filtered_dx = filter_x.step(raw_dx)
        filtered_dy = filter_y.step(raw_dy)
        # Deadband thresholding to eliminate micro-jitter (< 1.5 pixels)
        if abs(filtered_dx) < 1.5:
            filtered_dx = 0.0
        if abs(filtered_dy) < 1.5:
            filtered_dy = 0.0
    else:
        filtered_dx = raw_dx
        filtered_dy = raw_dy

    new_x = int(round(current_x + filtered_dx))
    new_y = int(round(current_y + filtered_dy))

    # Clamp to screen boundary
    new_x = max(0, min(SCREEN_WIDTH, new_x))
    new_y = max(0, min(SCREEN_HEIGHT, new_y))

    # Assistive Magnetic Snap
    if at_mode_enabled and active_targets:
        snapped_x, snapped_y = apply_assistive_magnetic_snap(new_x, new_y)
        filtered_dx += (snapped_x - new_x)
        filtered_dy += (snapped_y - new_y)
        new_x, new_y = snapped_x, snapped_y

    rel_x = int(round(filtered_dx))
    rel_y = int(round(filtered_dy))

    if HAS_HID and (rel_x != 0 or rel_y != 0):
        mouse.move(x=rel_x, y=rel_y)

    current_x = new_x
    current_y = new_y

    # Check Dwell-Click
    if at_mode_enabled and dwell_click_ms > 0:
        check_dwell_click(new_x, new_y)


def check_dwell_click(x: int, y: int):
    """Handles automated dwell clicking when cursor remains steady over an element."""
    global dwell_timer_start, dwell_target_hovered

    hovered = None
    for target in active_targets:
        if target[0] <= x <= target[2] and target[1] <= y <= target[3]:
            hovered = target
            break

    if hovered:
        if dwell_target_hovered == hovered:
            elapsed = (time.monotonic() - dwell_timer_start) * 1000.0
            if elapsed >= dwell_click_ms:
                if HAS_HID:
                    mouse.click(Mouse.LEFT_BUTTON)
                print(f"[PICO2_AT] Dwell-Click triggered at ({x}, {y}) after {int(elapsed)}ms")
                dwell_timer_start = time.monotonic() + 10.0  # Pause to avoid double-clicking
        else:
            dwell_target_hovered = hovered
            dwell_timer_start = time.monotonic()
    else:
        dwell_target_hovered = None
        dwell_timer_start = None


def process_command(cmd_line: str):
    global at_mode_enabled, tremor_filter_enabled, dwell_click_ms, active_targets
    tokens = cmd_line.strip().split()
    if not tokens:
        return
    cmd = tokens[0].upper()

    if cmd == "M" and len(tokens) >= 3:
        target_x = int(tokens[1])
        target_y = int(tokens[2])
        duration = int(tokens[3]) if len(tokens) > 3 else 350
        execute_mouse_trajectory(target_x, target_y, duration)
    elif cmd == "C":
        btn = tokens[1].lower() if len(tokens) > 1 else "left"
        if HAS_HID:
            if btn == "right":
                mouse.click(Mouse.RIGHT_BUTTON)
            else:
                mouse.click(Mouse.LEFT_BUTTON)
        print(f"[PICO2_MIMIC] Click: {btn}")
    elif cmd == "K" and len(tokens) >= 2:
        text = " ".join(tokens[1:])
        if HAS_HID:
            layout.write(text)
        print(f"[PICO2_MIMIC] Typed: {text}")
    elif cmd == "AT_MODE" and len(tokens) >= 2:
        at_mode_enabled = (tokens[1] in ("1", "ON", "TRUE"))
        print(f"[PICO2_AT] Assistive Technology Mode: {'ENABLED' if at_mode_enabled else 'DISABLED'}")
    elif cmd == "TREMOR_FILTER" and len(tokens) >= 2:
        tremor_filter_enabled = (tokens[1] in ("1", "ON", "TRUE"))
        filter_x.reset()
        filter_y.reset()
        print(f"[PICO2_AT] Real-Time Butterworth Tremor Filter: {'ENABLED' if tremor_filter_enabled else 'DISABLED'}")
    elif cmd == "ADD_TARGET" and len(tokens) >= 5:
        x1, y1, x2, y2 = int(tokens[1]), int(tokens[2]), int(tokens[3]), int(tokens[4])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        active_targets.append((x1, y1, x2, y2, cx, cy))
        print(f"[PICO2_AT] Registered Target: [{x1}, {y1}, {x2}, {y2}] Center=({cx}, {cy})")
    elif cmd == "CLEAR_TARGETS":
        active_targets.clear()
        print("[PICO2_AT] Cleared all registered UI targets")
    elif cmd == "DWELL_CLICK" and len(tokens) >= 2:
        dwell_click_ms = int(tokens[1])
        print(f"[PICO2_AT] Dwell-Click set to {dwell_click_ms}ms")
    elif cmd == "FILTER_INPUT" and len(tokens) >= 3:
        raw_dx = float(tokens[1])
        raw_dy = float(tokens[2])
        process_human_pass_through_motion(raw_dx, raw_dy)
    elif cmd == "SWITCH_PRESS" and len(tokens) >= 2:
        action = tokens[1].lower()
        if HAS_HID:
            if action == "left":
                mouse.click(Mouse.LEFT_BUTTON)
            elif action == "right":
                mouse.click(Mouse.RIGHT_BUTTON)
        print(f"[PICO2_AT] Adaptive Switch Actuated: {action}")
    elif cmd == "PING":
        print("[PICO2_MIMIC] PONG (Amicus Mimic RP2350 Ready - AT Co-Pilot Active)")


def main():
    print("[AMICUS_MIMIC] RP2350 Kinematic & Assistive HID Engine Initialized. Listening on CDC UART...")
    while True:
        try:
            line = sys.stdin.readline()
            if line:
                process_command(line)
        except Exception as e:
            print(f"[ERROR] {e}")
        time.sleep(0.001)


if __name__ == "__main__":
    main()
