#============================================================
# USER SETTINGS
# ============================================================

start_voltage = 4.0

SETPOINT = 120.0

RAMP_TIME = 20 * 60       # 20 minutes in seconds

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



# ============================================================
# OPEN PICO TC-08
# ============================================================

device = tc08.usb_tc08_open_unit()

if device <= 0:
    raise RuntimeError("Could not open Pico TC-08")

# 50 Hz mains rejection
tc08.usb_tc08_set_mains(device, 50)

# Channel 1, Type-T thermocouple
tc08.usb_tc08_set_channel(device, 1, ord('T'))

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

# Select heater channel
supply.write("INST:NSEL 2")

# Current limit
supply.write("CURR 0.6")

# Initial voltage
supply.write(f"VOLT {start_voltage:.3f}")

# Turn output on
supply.write("OUTPut ON")

print("Keysight PSU connected")
print(f"Starting voltage = {start_voltage:.2f} V")


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
    "PID_enabled"
])

print(f"Logging to {csv_filename}")


# ============================================================
# LIVE PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

line_temp, = ax.plot([], [], label="Temperature")
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
        return line_temp, line_voltage

    temp_c = float(temps[1])
  

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

    csv_writer.writerow([
        ts,
        elapsed_time,
        temp_c,
        temp_k,
        pid_voltage,
        ENABLE_PID
    ])
    csv_file.flush()

    # --------------------------------------------------------
    # Update plot
    # --------------------------------------------------------
    line_temp.set_data(xdata, ydata)
    line_voltage.set_data(xdata, pid_outputs)

    ax.relim()
    ax.autoscale_view()

    print(
        f"T = {temp_c:7.2f} °C | "
        f"V = {pid_voltage:5.2f} V | "
        f"PID = {ENABLE_PID}"
    )

    return line_temp, line_voltage

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

plt.show()
# ============================================================
# CLEANUP
# Run this cell when finished.
# ============================================================

try:
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