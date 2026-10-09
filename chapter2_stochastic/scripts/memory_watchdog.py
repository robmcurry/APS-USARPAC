"""
memory_watchdog.py

External, independent memory-pressure watchdog for long solver runs on this
machine -- reused from the design in the earlier 3-hour MIPFocus probe
(docs/individual_vehicle_lazy_subtour_report.md, Run 5 / "System memory
context"), where macOS's own jetsam killer had previously caused an
uncontrolled kill and a graceful external SIGTERM was used instead.

Samples, every --interval seconds (default 15s), independent of the target
process:
  - kern.memorystatus_vm_pressure_level (macOS system-wide pressure: 1=normal,
    2=warn, 4=critical)
  - the target PID's own RSS (via ps, same method as PeakRSSSampler)

Logs every sample to a CSV. If pressure is elevated (>1) for
--sustained-samples consecutive samples (default 8 -> 2 minutes at 15s), OR a
single critical (4) reading occurs, sends a graceful SIGTERM to the target
PID and logs the action, then exits. Does not touch the target process
otherwise -- this is a safety net, not a scheduler.

Run: python scripts/memory_watchdog.py --pid <PID> --csv <path> [--interval 15] [--sustained-samples 8]
"""
import argparse
import csv
import os
import signal
import subprocess
import sys
import time


def read_pressure_level() -> int:
    out = subprocess.check_output(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"])
    return int(out.strip())


def read_rss_mb(pid: int) -> float:
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)])
    return int(out.strip()) / 1024.0


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--sustained-samples", type=int, default=8)
    args = parser.parse_args()

    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["elapsed_sec", "pressure_level", "target_rss_mb", "action"])

        t0 = time.time()
        consecutive_elevated = 0
        print(f"[watchdog] monitoring PID {args.pid} every {args.interval}s "
              f"(sustained-elevated threshold: {args.sustained_samples} samples)")
        sys.stdout.flush()

        while True:
            if not pid_alive(args.pid):
                writer.writerow([f"{time.time() - t0:.1f}", "", "", "target_exited"])
                f.flush()
                print(f"[watchdog] target PID {args.pid} no longer running -- exiting")
                break

            try:
                pressure = read_pressure_level()
            except Exception:
                pressure = -1
            try:
                rss = read_rss_mb(args.pid)
            except Exception:
                rss = -1.0

            action = ""
            if pressure == 4:
                action = "SIGTERM_critical_pressure"
            elif pressure > 1:
                consecutive_elevated += 1
                if consecutive_elevated >= args.sustained_samples:
                    action = "SIGTERM_sustained_elevated_pressure"
            else:
                consecutive_elevated = 0

            elapsed = time.time() - t0
            writer.writerow([f"{elapsed:.1f}", pressure, f"{rss:.0f}", action])
            f.flush()

            if action:
                print(f"[watchdog] {action} at t={elapsed:.1f}s "
                      f"(pressure={pressure}, target_rss={rss:.0f}MB) "
                      f"-- sending graceful SIGTERM to PID {args.pid}")
                sys.stdout.flush()
                try:
                    os.kill(args.pid, signal.SIGTERM)
                except OSError:
                    pass
                writer.writerow([f"{time.time() - t0:.1f}", pressure, f"{rss:.0f}", "SIGTERM_sent"])
                f.flush()
                break

            time.sleep(args.interval)


if __name__ == "__main__":
    main()
