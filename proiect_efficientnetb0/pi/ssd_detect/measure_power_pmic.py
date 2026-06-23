import argparse
import csv
import os
import re
import signal
import statistics
import subprocess
import time
from datetime import datetime


RESULTS_DIR = "/home/bogdanavr/Desktop/pi/results"
SAMPLE_INTERVAL = 1.0


def run_cmd(command):
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def read_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r", encoding="utf-8") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return None


def read_throttled():
    out = run_cmd("vcgencmd get_throttled")
    # Example: throttled=0x0
    if "throttled=" in out:
        return out.split("throttled=")[-1].strip()
    return ""


def read_pmic():
    """
    Reads Raspberry Pi 5 PMIC ADC values using vcgencmd pmic_read_adc.

    Returns:
        estimated_power_w: sum of internal rail powers where both current and voltage exist
        ext5v_v: external 5V input voltage, if available
        rails: dictionary with rail powers
    """
    out = run_cmd("vcgencmd pmic_read_adc")

    currents = {}
    voltages = {}
    ext5v_v = None

    # Examples:
    # 3V3_SYS_A current(1)=0.06245952A
    # 3V3_SYS_V volt(9)=3.30632200V
    # EXT5V_V volt(24)=5.06788000V
    pattern = re.compile(
        r"^\s*([A-Za-z0-9_]+)\s+(current|volt)\(\d+\)=([0-9.]+)([AV])"
    )

    for line in out.splitlines():
        match = pattern.match(line)
        if not match:
            continue

        name, kind, value, unit = match.groups()
        value = float(value)

        if name == "EXT5V_V" and kind == "volt":
            ext5v_v = value
            continue

        if name.endswith("_A") and kind == "current":
            base = name[:-2]
            currents[base] = value


        elif name.endswith("_V") and kind == "volt":
            base = name[:-2]
            voltages[base] = value

    rails = {}
    for base in currents:
        if base in voltages:
            rails[base] = currents[base] * voltages[base]

    estimated_power_w = sum(rails.values())

    return estimated_power_w, ext5v_v, rails


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def start_target_command(cmd):
    if not cmd:
        return None

    # Start in a new process group so we can terminate the whole app cleanly.
    return subprocess.Popen(
        cmd,
        shell=True,
        preexec_fn=os.setsid,
    )


def stop_target_process(proc):
    if proc is None:
        return

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        time.sleep(2)
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass


def write_summary(summary_path, summary):
    file_exists = os.path.exists(summary_path)

    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, help="Example: idle, cpu_live, npu_live")
    parser.add_argument("--duration", type=float, default=60.0, help="Measurement duration in seconds")
    parser.add_argument("--cmd", default=None, help="Optional command to run during measurement")
    parser.add_argument("--sample_interval", type=float, default=SAMPLE_INTERVAL)
    args = parser.parse_args()

    ensure_results_dir()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples_csv = os.path.join(
        RESULTS_DIR,
        f"power_samples_{args.scenario}_{timestamp}.csv"
    )
    summary_csv = os.path.join(RESULTS_DIR, "power_summary.csv")

    print("========================================")
    print(f"Scenario: {args.scenario}")
    print(f"Duration: {args.duration} s")
    print(f"Sample interval: {args.sample_interval} s")
    print(f"Samples CSV: {samples_csv}")
    print(f"Summary CSV: {summary_csv}")
    print("========================================")


    proc = start_target_command(args.cmd)

    powers = []
    ext5v_values = []
    temp_values = []

    start = time.time()

    with open(samples_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "timestamp",
            "elapsed_s",
            "scenario",
            "estimated_pmic_power_w",
            "ext5v_v",
            "temp_c",
            "throttled",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        try:
            while True:
                now = time.time()
                elapsed = now - start

                if elapsed >= args.duration:
                    break

                power_w, ext5v_v, rails = read_pmic()
                temp_c = read_temp_c()
                throttled = read_throttled()

                powers.append(power_w)

                if ext5v_v is not None:
                    ext5v_values.append(ext5v_v)

                if temp_c is not None:
                    temp_values.append(temp_c)

                writer.writerow({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed_s": round(elapsed, 3),
                    "scenario": args.scenario,
                    "estimated_pmic_power_w": round(power_w, 4),
                    "ext5v_v": round(ext5v_v, 4) if ext5v_v is not None else "",
                    "temp_c": round(temp_c, 2) if temp_c is not None else "",
                    "throttled": throttled,
                })

                print(
                    f"{elapsed:6.1f}s | "
                    f"P_est={power_w:6.3f} W | "
                    f"EXT5V={ext5v_v if ext5v_v is not None else 0:5.3f} V | "
                    f"Temp={temp_c if temp_c is not None else 0:5.1f} C | "
                    f"Throttled={throttled}"
                )

                time.sleep(args.sample_interval)

        finally:
            stop_target_process(proc)

    if len(powers) == 0:
        print("No samples collected.")
        return

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scenario": args.scenario,
        "duration_s": args.duration,
        "samples": len(powers),
        "mean_estimated_pmic_power_w": round(statistics.mean(powers), 4),
        "min_estimated_pmic_power_w": round(min(powers), 4),
        "max_estimated_pmic_power_w": round(max(powers), 4),
        "mean_ext5v_v": round(statistics.mean(ext5v_values), 4) if ext5v_values else "",
        "min_ext5v_v": round(min(ext5v_values), 4) if ext5v_values else "",
        "mean_temp_c": round(statistics.mean(temp_values), 2) if temp_values else "",
        "max_temp_c": round(max(temp_values), 2) if temp_values else "",
    }

    write_summary(summary_csv, summary)

    print("\n=== SUMMARY ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("===============")


if __name__ == "__main__":
    main()




