# ============================================================
# IMPORTS
# ============================================================

import ctypes
import csv
import time
from datetime import datetime

import pyvisa
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from picosdk.usbtc08 import usbtc08 as tc08

# ADDED: imports for thermistor conversion.
import math
import numpy as np

# Measured terminal-board parameters at the input terminals in use.
# Scaling means: thermistor voltage = TC-08 raw voltage * 10.105.
# 10.1 kohm is the TOTAL effective input impedance of the connected board.
# Do not add the TC-08 input impedance separately to this measured value.
TERMINAL_BOARD_SCALE = 10.105
TERMINAL_BOARD_INPUT_OHM = 10.1e3

# G10K3976 nominal datasheet table, now including room temperature.
TEMPERATURE_C = np.arange(0.0, 151.0)
RESISTANCE_OHM = 1000.0 * np.array([
    32.68, 31.06, 29.53, 28.08, 26.71, 25.41, 24.19, 23.03,
    21.93, 20.90, 19.91, 18.98, 18.10, 17.26, 16.47, 15.72,
    15.00, 14.33, 13.68, 13.07, 12.49, 11.94, 11.42, 10.92,
    10.45, 10.00, 9.572, 9.164, 8.776, 8.407, 8.055, 7.720,
    7.401, 7.097, 6.806, 6.530, 6.266, 6.014, 5.773, 5.544,
    5.325, 5.115, 4.915, 4.724, 4.542, 4.367, 4.200, 4.041,
    3.888, 3.742,
    3.602, 3.468, 3.339, 3.217, 3.099, 2.986, 2.878, 2.774,
    2.675, 2.580, 2.488, 2.401, 2.317, 2.236, 2.158, 2.084,
    2.012, 1.944, 1.878, 1.814, 1.753, 1.695, 1.639, 1.584,
    1.532, 1.482, 1.434, 1.387, 1.343, 1.300, 1.258, 1.218,
    1.180, 1.143, 1.107, 1.073, 1.039, 1.007, 0.9766, 0.9468,
    0.9181, 0.8904, 0.8637, 0.8379, 0.8130, 0.7890, 0.7658,
    0.7434, 0.7217, 0.7008, 0.6806, 0.6610, 0.6421, 0.6239,
    0.6062, 0.5892, 0.5727, 0.5567, 0.5413, 0.5263, 0.5119,
    0.4979, 0.4843, 0.4712, 0.4585, 0.4462, 0.4343, 0.4227,
    0.4115, 0.4007, 0.3902, 0.3800, 0.3702, 0.3606, 0.3513,
    0.3423, 0.3336, 0.3252, 0.3169, 0.3090, 0.3013, 0.2938,
    0.2865, 0.2794, 0.2725, 0.2659, 0.2594, 0.2531, 0.2470,
    0.2411, 0.2353, 0.2297, 0.2243, 0.2190, 0.2139, 0.2089,
    0.2040, 0.1993, 0.1947, 0.1902, 0.1859,
])


def convert_ch2(
    voltage_mv, overflow_flags=0, excitation_a=25e-6,
    scale_factor=TERMINAL_BOARD_SCALE,
    terminal_input_ohm=TERMINAL_BOARD_INPUT_OHM,
):
    """Return resistance in ohms, temperature in Celsius, and a status string.

    Raw TC-08 voltage-mode values are mV, AFTER terminal-board attenuation.
    The +/-70 mV range check applies to that raw reading, before scaling.
    Channel 2 overflow is bit 2 in the get_single overflow word.

    Assumes the total fixed excitation current splits between the thermistor
    and the terminal-board input connected in parallel across its sense leads:
        sensor_v = raw_mv * 1e-3 * scale_factor
        thermistor_current = excitation_a - sensor_v / terminal_input_ohm
        resistance = sensor_v / thermistor_current
    Input impedance and scale must describe the same connected board/range.
    Invalid results remain NaN; the nominal table does not calibrate the sensor.
    """
    excitation_a = float(excitation_a)
    if not math.isfinite(excitation_a) or excitation_a <= 0:
        raise ValueError('Excitation current must be positive and finite.')
    scale_factor = float(scale_factor)
    terminal_input_ohm = float(terminal_input_ohm)
    if not math.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError('Terminal-board scaling factor must be positive and finite.')
    if not math.isfinite(terminal_input_ohm) or terminal_input_ohm <= 0:
        raise ValueError('Terminal-board input impedance must be positive and finite.')
    voltage_mv = float(voltage_mv)
    if not math.isfinite(voltage_mv):
        return math.nan, math.nan, 'nonfinite_voltage'
    if int(overflow_flags) & (1 << 2) or abs(voltage_mv) >= 70.0:
        return math.nan, math.nan, 'channel_2_overrange'
    if voltage_mv <= 0:
        return math.nan, math.nan, 'nonpositive_voltage_check_polarity'
    sensor_voltage_v = voltage_mv * 1e-3 * scale_factor
    thermistor_current_a = excitation_a - sensor_voltage_v / terminal_input_ohm
    if thermistor_current_a <= 0:
        return math.nan, math.nan, 'nonpositive_thermistor_current_check_scale_or_wiring'
    resistance = sensor_voltage_v / thermistor_current_a
    r_min, r_max = RESISTANCE_OHM[-1], RESISTANCE_OHM[0]
    tolerance = 1e-9 * r_max
    if not r_min - tolerance <= resistance <= r_max + tolerance:
        return resistance, math.nan, 'outside_0_to_150_C_table'
    # Only clamp machine-rounding excursions at table endpoints.
    resistance_for_interp = float(np.clip(resistance, r_min, r_max))
    inverse_kelvin = np.interp(
        np.log(resistance_for_interp),
        np.log(RESISTANCE_OHM[::-1]),
        1.0 / (TEMPERATURE_C[::-1] + 273.15),
    )
    return resistance, float(1.0 / inverse_kelvin - 273.15), 'ok'


#============================================================
# USER SETTINGS
# ============================================================

start_voltage = 4.0

# Fixed auxiliary supply on Keysight output 3 (independent of the heater PID).
output_3_voltage = 9.0

SETPOINT = 120.0

RAMP_TIME = 15 * 60       # 20 minutes in seconds

initial_temperature = None
ramp_start_time = None

KP = 0.15
KI = 0.001
KD = 0

max_volt = 6.8
ENABLE_PID = False

manual_voltage = start_voltage
voltage_step = 0.1

dt = 0.5

# ADDED: externally supplied constant thermistor excitation current.
excitation_current = 25e-6

# ============================================================
# OPEN PICO TC-08
# ============================================================

device = tc08.usb_tc08_open_unit()

if device <= 0:
    raise RuntimeError("Could not open Pico TC-08")

# 50 Hz mains rejection: SDK uses 0 for 50 Hz, 1 for 60 Hz.
tc08.usb_tc08_set_mains(device, 0)

# Channel 1, Type-T thermocouple
tc08.usb_tc08_set_channel(device, 1, ord('T'))

# ADDED: Channel 2, voltage input; the SDK returns millivolts.
if tc08.usb_tc08_set_channel(device, 2, ord('X')) != 1:
    tc08.usb_tc08_close_unit(device)
    raise RuntimeError("Could not configure Pico channel 2 as a voltage input")

temps = (ctypes.c_float * 9)()
overflow = ctypes.c_int16()

print("Pico TC-08 connected")

# ============================================================
# OPEN KEYSIGHT POWER SUPPLY
# ============================================================

rm = pyvisa.ResourceManager()

supply = rm.open_resource(
    "USB0::10893::4610::MY61002725::0::INSTR"
)

# Configure and enable the fixed-voltage output.
# Output 3 retains its existing current-limit setting.
supply.write("INST:NSEL 3")
supply.write(f"VOLT {output_3_voltage:.3f}")
supply.write("OUTPut ON")

# Select heater channel; subsequent PID/manual writes act on output 2.
supply.write("INST:NSEL 2")

# Current limit
supply.write("CURR 0.6")

# Initial voltage
supply.write(f"VOLT {start_voltage:.3f}")

# Turn output on
supply.write("OUTPut ON")

print("Keysight PSU connected")
print(f"Starting voltage = {start_voltage:.2f} V")
print(f"Output 3 fixed voltage = {output_3_voltage:.2f} V")
print(f"CH2 terminal board: scale = {TERMINAL_BOARD_SCALE:g}, "
      f"input impedance = {TERMINAL_BOARD_INPUT_OHM:g} ohm")

# ============================================================
# PID / DATA INITIALISATION
# ============================================================

integral = 0.0
last_error = 0.0
pid_just_enabled = False

start_time = time.time()

xdata = []
ydata = []
pid_outputs = []
ydata_out_of_loop = []  # ADDED

csv_filename = (
    "temperature_PID_" +
    datetime.now().strftime("%Y%m%d_%H%M%S") +
    ".csv"
)

csv_file = open(csv_filename, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "timestamp",
    "elapsed_time_s",
    "temperature_C",
    "temperature_K",
    "voltage_V",
    "PID_enabled",
    # ADDED: channel 2 measurement columns.
    "out_of_loop_temperature_C",
    "out_of_loop_voltage_mV",
    "out_of_loop_resistance_ohm",
    "out_of_loop_status",
    "out_of_loop_sensor_voltage_mV",
    "out_of_loop_thermistor_current_uA",
    "excitation_current_A",
    "terminal_board_scale",
    "terminal_board_input_ohm"
])

print(f"Logging to {csv_filename}")

# ============================================================
# LIVE PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

line_temp, = ax.plot([], [], label="Temperature")
# ADDED: out-of-loop temperature on the same temperature axis.
line_out_of_loop, = ax.plot([], [], label="Out-of-loop temperature")
ax.axhline(SETPOINT, linestyle="--", label="Setpoint")

ax.set_xlabel("Time (s)")
ax.set_ylabel("Temperature (°C)")
ax.grid(True)
ax.legend(loc="upper left")

ax_voltage = ax.twinx()
line_voltage, = ax_voltage.plot([], [], label="Voltage")
ax_voltage.set_ylabel("Voltage (V)")
ax_voltage.set_ylim(0, max_volt * 1.1)
ax_voltage.legend(loc="upper right")

# ============================================================
# TEMPERATURE READ + PID UPDATE
# ============================================================

def update(frame):
    global integral
    global last_error
    global ENABLE_PID
    global pid_just_enabled
    global manual_voltage
    global initial_temperature
    global ramp_start_time
    # Read TC-08 once
    status = tc08.usb_tc08_get_single(
        device,
        temps,
        ctypes.byref(overflow),
        0
    )

    if status == 0:
        print("TC-08 read failed")
        return line_temp, line_out_of_loop, line_voltage

    temp_c = float(temps[1])

    # CH2 raw mV -> undo attenuation -> subtract board input current -> R -> T.
    out_of_loop_voltage_mv = float(temps[2])
    out_of_loop_sensor_voltage_mv = out_of_loop_voltage_mv * TERMINAL_BOARD_SCALE
    out_of_loop_thermistor_current_ua = (
        excitation_current
        - out_of_loop_sensor_voltage_mv * 1e-3 / TERMINAL_BOARD_INPUT_OHM
    ) * 1e6
    out_of_loop_resistance, out_of_loop_temp_c, out_of_loop_status = convert_ch2(
        out_of_loop_voltage_mv,
        overflow.value,
        excitation_current,
        scale_factor=TERMINAL_BOARD_SCALE,
        terminal_input_ohm=TERMINAL_BOARD_INPUT_OHM,
    )

# Initialise ramp on first temperature reading
    if initial_temperature is None:
        initial_temperature = temp_c
        ramp_start_time = time.time()

    temp_k = temp_c + 273.15
    current_time = time.time()
    elapsed_time = current_time - start_time

    ramp_elapsed = time.time() - ramp_start_time

    ramp_fraction = min(ramp_elapsed / RAMP_TIME, 1.0)

    current_setpoint = (
        initial_temperature
        + ramp_fraction * (SETPOINT - initial_temperature)
    )

    # --------------------------------------------------------
    # PID
    # --------------------------------------------------------

    error = current_setpoint - temp_c

    if ENABLE_PID:
        error = current_setpoint - temp_c

        # Avoid a derivative kick on the first PID iteration
        if pid_just_enabled:
            last_error = error
            pid_just_enabled = False

        integral += error * dt
        derivative = (error - last_error) / dt

        pid_voltage = (
            KP * error
            + KI * integral
            + KD * derivative
        )

        pid_voltage = max(0.0, min(pid_voltage, max_volt))

        supply.write(f"VOLT {pid_voltage:.3f}")
        manual_voltage = pid_voltage

        last_error = error

    else:
        pid_voltage = manual_voltage
        supply.write(f"VOLT {pid_voltage:.3f}")

    # --------------------------------------------------------
    # Store / log
    # --------------------------------------------------------
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    xdata.append(elapsed_time)
    ydata.append(temp_c)
    pid_outputs.append(pid_voltage)
    ydata_out_of_loop.append(out_of_loop_temp_c)  # ADDED

    csv_writer.writerow([
        ts,
        elapsed_time,
        temp_c,
        temp_k,
        pid_voltage,
        ENABLE_PID,
        # ADDED
        out_of_loop_temp_c,
        out_of_loop_voltage_mv,
        out_of_loop_resistance,
        out_of_loop_status,
        out_of_loop_sensor_voltage_mv,
        out_of_loop_thermistor_current_ua,
        excitation_current,
        TERMINAL_BOARD_SCALE,
        TERMINAL_BOARD_INPUT_OHM
    ])
    csv_file.flush()

    # --------------------------------------------------------
    # Update plot
    # --------------------------------------------------------
    line_temp.set_data(xdata, ydata)
    line_out_of_loop.set_data(xdata, ydata_out_of_loop)  # ADDED
    line_voltage.set_data(xdata, pid_outputs)

    ax.relim()
    ax.autoscale_view()

    print(
        f"T = {temp_c:7.2f} °C | "
        f"Out-of-loop = {out_of_loop_temp_c:8.4f} °C ({out_of_loop_status}) | "
        f"CH2 raw = {out_of_loop_voltage_mv:.6f} mV | "
        f"Sensor = {out_of_loop_sensor_voltage_mv:.6f} mV | "
        f"Thermistor I = {out_of_loop_thermistor_current_ua:.4f} uA | "
        f"Overflow = {overflow.value:#011b} | "
        f"V = {pid_voltage:5.2f} V | "
        f"PID = {ENABLE_PID}"
    )

    return line_temp, line_out_of_loop, line_voltage

# ============================================================
# KEYBOARD CONTROL
# ============================================================

def on_key_press(event):
    global ENABLE_PID
    global pid_just_enabled
    global manual_voltage
    global integral
    global last_error

    if event.key == "p":
        ENABLE_PID = not ENABLE_PID

        if ENABLE_PID:
            pid_just_enabled = True
            integral = 0.0
            last_error = 0.0
            print("PID ENABLED")
        else:
            print("PID DISABLED")

    elif event.key == "up" and not ENABLE_PID:
        manual_voltage = min(manual_voltage + voltage_step, max_volt)
        supply.write(f"VOLT {manual_voltage:.3f}")
        print(f"Manual voltage = {manual_voltage:.2f} V")

    elif event.key == "down" and not ENABLE_PID:
        manual_voltage = max(manual_voltage - voltage_step, 0.0)
        supply.write(f"VOLT {manual_voltage:.3f}")
        print(f"Manual voltage = {manual_voltage:.2f} V")

fig.canvas.mpl_connect("key_press_event", on_key_press)

# ============================================================
# RUN
# ============================================================

ani = FuncAnimation(
    fig,
    update,
    interval=500,
    cache_frame_data=False
)

print("Controls:")
print("  p          : toggle PID")
print("  up arrow   : increase voltage in manual mode")
print("  down arrow : decrease voltage in manual mode")

plt.show(block=True)
# ============================================================
# CLEANUP
# Run this cell when finished.
# ============================================================

for output_channel in (2, 3):
    try:
        supply.write(f"INST:NSEL {output_channel}")
        supply.write("VOLT 0")
        supply.write("OUTPut OFF")
    except Exception:
        pass

try:
    csv_file.close()
except Exception:
    pass

try:
    tc08.usb_tc08_close_unit(device)
except Exception:
    pass

try:
    supply.close()
    rm.close()
except Exception:
    pass

print("Closed TC-08 and power supply")
