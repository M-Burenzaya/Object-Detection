import time
import os

from su.common.io.plc_memory import PLCMemory

# === Configuration ===
PLC_IP = '192.168.0.10'
PLC_PORT = 5000
START_ADDRESS = 0       # D0
WORD_COUNT = 10         # D0 to D9
DEVICE = 'D'
LOOP_COUNT = 10000

plc = PLCMemory(ipAddr=PLC_IP, port=PLC_PORT)

def connect():
    if plc.Connect() == 0:
        print("Connected to PLC")
    else:
        print("Could not connect to PLC")
        exit()

def print_values(values, base_addr, clear_lines=True):
    if clear_lines:
        print(f"\033[{len(values)}A", end="")  # Move up to overwrite

    for i, val in enumerate(values):
        print(f"  D{base_addr + i} = {val:5}", flush=True)

def batch_increment():
    current, ret = plc.ReadSerial(DEVICE, START_ADDRESS, WORD_COUNT)
    if ret != 0:
        print("Initial read failed.")
        return False

    new_vals = [(v + 1) % 65536 for v in current]
    ret = plc.WriteSerial(DEVICE, START_ADDRESS, WORD_COUNT, new_vals)
    if ret != 0:
        print("Write failed.")
        return False

    confirmed, ret = plc.ReadSerial(DEVICE, START_ADDRESS, WORD_COUNT)
    if ret != 0:
        print("Confirmation read failed.")
        return False

    print_values(confirmed, START_ADDRESS, clear_lines=True)
    return True

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\033[H\033[J", end="")  # ANSI fallback

if __name__ == "__main__":
    connect()
    clear_console()
    # print("Confirmed new values:\n")
    start_time = time.time()

    try:
        for i in range(LOOP_COUNT):
            success = batch_increment()
            if not success:
                break
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        total_time = time.time() - start_time
        avg_time = total_time / LOOP_COUNT
        print(f"\nCompleted {LOOP_COUNT} loops.")
        print(f"Total time: {total_time:.2f} sec")
        print(f"Average time per cycle: {avg_time*1000:.2f} ms")
        plc.Close()
        print("Connection closed.")
