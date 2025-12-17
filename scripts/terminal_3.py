
import simpy
import numpy as np
import pandas as pd
import time
from statistics import mean

# ------------------- PARAMETERS -------------------
RANDOM_SEED = 42
RUN_TIME = 1440  # minutes (24 hours)
P_DIGI = 0.19
P_DOMESTIC = 0.63       # probability passenger is domestic at security

CAP_GATE_REG = 16
CAP_GATE_DIGI = 6
CAP_CHECKIN = 75
CAP_SECURITY = 64
CAP_BOARDING = 50
NUM_GATES = 62

np.random.seed(RANDOM_SEED)
start_time = time.time()

# ---------- Terminal3-specific probabilities ----------
P_INT_FIRST = 0.10

# ------------------- TIME INTERVALS -------------------
time_blocks = np.array([
    0, 60, 120, 180, 240, 300, 315, 330, 345, 360, 375, 390, 405,
    420, 435, 450, 465, 480, 495, 510, 525, 540, 600, 660, 720, 780,
    840, 900, 960, 975, 990, 1005, 1020, 1035, 1050, 1065, 1080,
    1095, 1110, 1125, 1140, 1155, 1170, 1185, 1200, 1260, 1320, 1380
])

# ------------------- SERVICE TIME ARRAYS -------------------
service_gate = np.array([
    0.6313398271, 0.6687787477, 0.6572911588, 0.6669968642, 0.675445933, 0.6894550604,
    0.6669968642, 0.6529900573, 0.6678939253, 0.661342195, 0.5564363681, 0.5408726188,
    0.5667601856, 0.5582331446, 0.5615808828, 0.5459440065, 0.5607728805, 0.5525648611,
    0.5535697277, 0.5706285908, 0.5607728805, 0.5780896302, 0.5351558127, 0.5381025232,
    0.619551916, 0.629668227, 0.6638468631, 0.6350449612, 0.6287250867, 0.6538848841,
    0.6399820735, 0.6333044569, 0.629668227, 0.6315129914, 0.6305974233, 0.6430557696,
    0.629668227, 0.6254049224, 0.6379391479, 0.6562311305, 0.6553887898, 0.6536523263,
    0.6169922097, 0.6313398271, 0.6238171069, 0.6340802719, 0.6254049224, 0.6379391479
])

service_checkin = np.array([
    2.415264806, 2.710918307, 2.759280499, 2.690217507, 2.684573268, 2.69566418,
    2.710918307, 2.724725597, 2.745044548, 2.724725597, 2.142850389, 2.147481033,
    2.171740491, 2.204813156, 2.19413588, 2.18199644, 2.202266172, 2.196931411,
    2.188267191, 2.168070014, 2.160307638, 2.132986495, 2.151926982, 2.147481033,
    2.466513542, 2.447419847, 2.483835843, 2.507009228, 2.396351798, 2.437119776,
    2.447419847, 2.457204988, 2.514080824, 2.527365332, 2.483835843, 2.550954592,
    2.533612721, 2.516081133, 2.496551183, 2.520546397, 2.524861739, 2.558042493,
    2.575724723, 2.567252783, 2.520546397, 2.491203285, 2.406294185, 2.496551183
])

service_domestic_security = np.array([
    1.722883499, 1.93063134, 1.93063134, 1.956965733, 2.002711293, 2.073943637,
    2.073943637, 1.956965733, 2.02272483, 2.058167496, 1.624965569, 1.66792557,
    1.588720997, 1.624965569, 1.66792557, 1.735951714, 1.710920455, 1.640591446,
    1.640591446, 1.701356473, 1.719832046, 1.607762512, 1.640591446, 1.719832046,
    1.993722175, 1.993722175, 1.980176757, 1.932793659, 1.980176757, 1.893992441,
    1.980176757, 1.949851224, 2.029161916, 1.893992441, 1.914245011, 1.871778783,
    1.893992441, 1.894360551, 1.894360551, 1.953212103, 1.997385199, 1.965389414,
    1.940136172, 2.03929951, 2.101184344, 1.835339077, 1.642278962, 1.722883499
])

service_international_security = np.array([
    2.20586958, 2.030187511, 2.105513459, 2.105513459, 2.105513459, 2.105513459,
    2.105513459, 2.030187511, 2.030187511, 2.030187511, 2.343263937, 2.343263937,
    2.343263937, 2.343263937, 2.343263937, 2.343263937, 2.343263937, 2.241884224,
    2.343263937, 2.343263937, 2.343263937, 2.343263937, 2.241884224, 2.343263937,
    2.464934502, 2.464934502, 2.464934502, 2.536452407, 2.595776007, 2.595776007,
    2.464934502, 2.595776007, 2.464934502, 2.464934502, 2.464934502, 2.464934502,
    2.595776007, 2.20586958, 2.119774699, 2.26546552, 2.20586958, 2.20586958,
    2.20586958, 2.26546552, 2.119774699, 2.119774699, 2.119774699, 2.20586958
])

# ---------- Helper ----------
def get_service_time(service_array, t):
    idx = np.searchsorted(time_blocks, t, side="right") - 1
    idx = min(max(idx, 0), len(service_array) - 1)
    # sample an exponential with mean = service_array[idx]
    return np.random.exponential(service_array[idx])

def D_gate_Digi(): return 0.5 + np.random.exponential(0.05)
def D_gate_Reg(t): return get_service_time(service_gate, t)
def D_checkin(t):  return get_service_time(service_checkin, t)
def D_security_domestic(t): return get_service_time(service_domestic_security, t)
# security functions now accept current time
def D_security_int_first(t): return get_service_time(service_international_security, t)
def D_security_int_econ(t):  return get_service_time(service_international_security, t)
def D_walking():      return np.random.triangular(5.0, 10.0, 10.0)
def D_waiting():      return np.random.uniform(20.0, 40.0)
def D_boarding():     return np.random.triangular(5.0, 8.0, 10.0)

# ------------------- DATA -------------------
df = pd.read_csv('data/Terminal3 Flight Data.csv', skipinitialspace=True)
# Clean up: remove empty columns and ensure proper data types
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns = df.columns.str.strip()
df['Dep_time_min'] = pd.to_numeric(df['Dep_time_min'], errors='coerce').fillna(0)
df['N_passengers'] = pd.to_numeric(df['N_passengers'], errors='coerce').fillna(0).astype(int)
df['Airline'] = df['Airline'].astype(str).str.strip()
if 'Type' in df.columns:
    df['Type'] = df['Type'].astype(str).str.strip()

# ---------- Initialize Metrics ----------
cycle_times = []
total_entered = 0     # ✅ added
waits = {
    "gate_reg": [], "gate_digi": [],
    "checkin": [],
    "security_domestic": [], "security_int_first": [], "security_int_econ": [],
    "waiting_hall": [], "boarding": []
}
queue_lengths = {
    "checkin": [], "security_domestic": [], "security_international": [], "boarding_domestic": [], "boarding_international": []
}
busy_time = {"checkin": 0.0, "security_domestic": 0.0, "security_int_first": 0.0, "security_int_econ": 0.0, "boarding_domestic": 0.0, "boarding_international": 0.0}
throughput = np.zeros(28, dtype=int)
results_by_type = {"domestic": [], "int_first": [], "int_econ": []}
system_count = []

num_gates_domestic = max(1, int(round(NUM_GATES * P_DOMESTIC)))
num_gates_international = NUM_GATES - num_gates_domestic

# ---------- Passenger Process ----------
def passenger(env, pid, dep_time, ptype, gate_reg, gate_digi, checkins, securities_dom, securities_int, boardings_dom, boardings_int):
    t0 = env.now
    system_count.append((env.now, +1))

    if np.random.rand() < P_DIGI:
        chosen = min(gate_digi, key=lambda r: len(r.queue))
        with chosen.request() as req:
            t_req = env.now; yield req
            waits["gate_digi"].append(env.now - t_req)
            yield env.timeout(D_gate_Digi())
    else:
        chosen = min(gate_reg, key=lambda r: len(r.queue))
        with chosen.request() as req:
            t_req = env.now; yield req
            waits["gate_reg"].append(env.now - t_req)
            yield env.timeout(D_gate_Reg(env.now))

    chosen = min(checkins, key=lambda r: len(r.queue))
    queue_lengths["checkin"].append(len(chosen.queue))
    with chosen.request() as req:
        t_req = env.now; yield req
        waits["checkin"].append(env.now - t_req)
        st = D_checkin(env.now)
        yield env.timeout(st)
        busy_time["checkin"] += st

    if ptype == "D":
        chosen = min(securities_dom, key=lambda r: len(r.queue))
        queue_lengths["security_domestic"].append(len(chosen.queue))
        with chosen.request() as req:
            t_req = env.now; yield req
            waits["security_domestic"].append(env.now - t_req)
            st = D_security_domestic(env.now)
            yield env.timeout(st)
            busy_time["security_domestic"] += st

        chosen_board = min(boardings_dom, key=lambda r: len(r.queue))
        queue_lengths["boarding_domestic"].append(len(chosen_board.queue))
        with chosen_board.request() as req:
            t_req = env.now; yield req
            waits["boarding"].append(env.now - t_req)
            st2 = D_boarding()
            yield env.timeout(st2)
            busy_time["boarding_domestic"] += st2
        results_by_type["domestic"].append(env.now - t0)

    elif ptype == "I":
        if np.random.rand() < 0.2:
            chosen = min(securities_int, key=lambda r: len(r.queue))
            queue_lengths["security_international"].append(len(chosen.queue))
            with chosen.request() as req:
                t_req = env.now; yield req
                waits["security_int_first"].append(env.now - t_req)
                st = D_security_int_first(env.now)
                yield env.timeout(st)
                busy_time["security_int_first"] += st

            chosen_board = min(boardings_int, key=lambda r: len(r.queue))
            queue_lengths["boarding_international"].append(len(chosen_board.queue))
            with chosen_board.request() as req:
                t_req = env.now; yield req
                waits["boarding"].append(env.now - t_req)
                st2 = D_boarding()
                yield env.timeout(st2)
                busy_time["boarding_international"] += st2
            results_by_type["int_first"].append(env.now - t0)
        else:
            chosen = min(securities_int, key=lambda r: len(r.queue))
            queue_lengths["security_international"].append(len(chosen.queue))
            with chosen.request() as req:
                t_req = env.now; yield req
                waits["security_int_econ"].append(env.now - t_req)
                st = D_security_int_econ(env.now)
                yield env.timeout(st)
                busy_time["security_int_econ"] += st

            chosen_board = min(boardings_int, key=lambda r: len(r.queue))
            queue_lengths["boarding_international"].append(len(chosen_board.queue))
            with chosen_board.request() as req:
                t_req = env.now; yield req
                waits["boarding"].append(env.now - t_req)
                st2 = D_boarding()
                yield env.timeout(st2)
                busy_time["boarding_international"] += st2
            results_by_type["int_econ"].append(env.now - t0)

    waiting_time = D_waiting()
    waits["waiting_hall"].append(waiting_time)
    yield env.timeout(D_walking() + waiting_time)

    cycle_times.append(env.now - t0)
    idx = min(int(env.now // 60), len(throughput) - 1)
    throughput[idx] += 1
    system_count.append((env.now, -1))

# ---------- Flight Source ----------
def flight_source(env, gate_reg, gate_digi, checkin_list,
                  securities_dom, securities_int, boardings_dom, boardings_int):
    """
    Generate passengers for each flight in the dataset.
    Each passenger arrives 60–120 minutes before the flight departure,
    with exponential inter-arrival times (realistic clustering).
    """
    global total_entered
    pid = 0
    eps = 1e-6  # tiny epsilon to avoid exact RUN_TIME

    for _, row in df.iterrows():
        dep_time = float(row["Dep_time_min"])
        n_passengers = int(row["N_passengers"])
        ptype = str(row["Type"]).strip().upper()

        # passengers arrive in the window [dep_time-120, dep_time-60]
        arr_start = max(0.0, dep_time - 120.0)
        arr_end = min(RUN_TIME - eps, dep_time - 60.0)

        # if invalid window (arr_end < arr_start), fix it
        if arr_end < arr_start:
            arr_end = arr_start

        if n_passengers <= 0:
            continue

        # total available time window for arrivals
        total_window = arr_end - arr_start
        if total_window <= 0:
            continue

        # --- Exponential inter-arrival times ---
        # mean inter-arrival time = total_window / n_passengers
        mean_arrival_gap = total_window / n_passengers
        inter_arrivals = np.random.exponential(scale=mean_arrival_gap, size=n_passengers)

        # normalize so arrivals fit inside window
        arrival_times = arr_start + np.cumsum(inter_arrivals)
        arrival_times = np.clip(arrival_times, arr_start, arr_end)
        arrival_times.sort()

        # --- Generate passengers at these times ---
        for arr_time in arrival_times:
            yield env.timeout(max(0, arr_time - env.now))
            pid += 1
            total_entered += 1
            env.process(
                passenger(
                    env, pid, dep_time, ptype,
                    gate_reg, gate_digi,
                    checkin_list, securities_dom, securities_int,
                    boardings_dom, boardings_int
                )
            )



# ---------- Simulation set-up ----------
env = simpy.Environment()
gate_reg = [simpy.Resource(env, capacity=1) for _ in range(CAP_GATE_REG)]
gate_digi = [simpy.Resource(env, capacity=1) for _ in range(CAP_GATE_DIGI)]
checkins = [simpy.Resource(env, capacity=1) for _ in range(CAP_CHECKIN)]
sec_dom_count = max(1, int(round(CAP_SECURITY * P_DOMESTIC)))
sec_int_count = max(1, CAP_SECURITY - sec_dom_count)
securities_dom = [simpy.Resource(env, capacity=1) for _ in range(sec_dom_count)]
securities_int = [simpy.Resource(env, capacity=1) for _ in range(sec_int_count)]
boardings_dom = [simpy.Resource(env, capacity=CAP_BOARDING) for _ in range(num_gates_domestic)]
boardings_int = [simpy.Resource(env, capacity=CAP_BOARDING) for _ in range(num_gates_international)]
env.process(flight_source(env, gate_reg, gate_digi, checkins, securities_dom, securities_int, boardings_dom, boardings_int))

print("\n--- Simulation Started (Terminal 3) ---\n")
env.run(until=RUN_TIME+300)

# ------------------ ARRIVAL RATE PER HOUR ------------------
# Extract only the passenger entry events (+1)
arrival_times = [t for t, delta in system_count if delta == +1]

# Convert timestamps (minutes) → hour index (0–27)
arrival_hours = np.clip((np.array(arrival_times) // 60).astype(int), 0, 27)

# Count arrivals by hour
arrival_rate = np.bincount(arrival_hours, minlength=28)

print("\n🚶 Passenger Arrival Rate Per Hour:")
for h in range(len(arrival_rate)):
    print(f"{h:02d}:00 - {arrival_rate[h]} passengers")


print("\n--- Simulation Completed ---")
print(f"Total runtime: {round(time.time() - start_time, 2)} seconds")

# ---------- Metrics ----------
def avg(lst): return mean(lst) if lst else 0.0
def safe_max(lst): return float(np.max(lst)) if len(lst) else 0.0

avg_queue_len = {k: np.mean(v) if len(v) else 0.0 for k, v in queue_lengths.items()}
max_queue_len = {k: safe_max(v) for k, v in queue_lengths.items()}

utilization = {}
utilization["checkin"] = busy_time["checkin"] / (len(checkins) * RUN_TIME) if len(checkins) else 0.0
utilization["security_domestic"] = busy_time["security_domestic"] / (len(securities_dom) * RUN_TIME) if len(securities_dom) else 0.0
utilization["security_int_first"] = busy_time["security_int_first"] / (len(securities_int) * RUN_TIME) if len(securities_int) else 0.0
utilization["security_int_econ"] = busy_time["security_int_econ"] / (len(securities_int) * RUN_TIME) if len(securities_int) else 0.0
utilization["boarding_domestic"] = busy_time["boarding_domestic"] / (len(boardings_dom) * RUN_TIME) if len(boardings_dom) else 0.0
utilization["boarding_international"] = busy_time["boarding_international"] / (len(boardings_int) * RUN_TIME) if len(boardings_int) else 0.0

percentiles = np.percentile(cycle_times, [5, 50, 95]) if cycle_times else [0, 0, 0]
system_trend = np.cumsum([c for _, c in sorted(system_count, key=lambda x: x[0])]) if system_count else np.array([0])
avg_system = np.mean(system_trend) if len(system_trend) else 0.0
max_system = np.max(system_trend) if len(system_trend) else 0.0

# ---------- Additional stats ----------
waiting_gate_reg = avg(waits["gate_reg"])
waiting_gate_digi = avg(waits["gate_digi"])
waiting_checkin = avg(waits["checkin"])
waiting_sec_dom = avg(waits["security_domestic"])
waiting_sec_int_first = avg(waits["security_int_first"])
waiting_sec_int_econ = avg(waits["security_int_econ"])

waiting_hall_avg = avg(waits["waiting_hall"])
waiting_hall_max = max(waits["waiting_hall"]) if waits["waiting_hall"] else 0.0
waiting_hall_95 = np.percentile(waits["waiting_hall"], 95) if waits["waiting_hall"] else 0.0

boarding_queue_avg = avg(waits["boarding"])

# throughput per hour strings
throughput_lines = []
for hr in range(28):
    count = int(throughput[hr])
    throughput_lines.append((hr, count))

# breakdown by type
breakdown = {}
for k in results_by_type.keys():
    arr = np.array(results_by_type[k]) if results_by_type[k] else np.array([])
    cnt = len(arr)
    avg_t = float(np.mean(arr)) if cnt else 0.0
    pct95 = float(np.percentile(arr, 95)) if cnt else 0.0
    breakdown[k] = {"count": cnt, "avg": avg_t, "95th": pct95}

# ---------- Print results ----------
print(f"\nTotal passengers entered: {total_entered}")
print(f"Total passengers completed: {len(cycle_times)}")
print(f"Passengers still in system (not completed): {total_entered - len(cycle_times)}")

if cycle_times:
    print(f"\nAverage total time in system: {np.mean(cycle_times):.2f} min")
    print(f"Min: {np.min(cycle_times):.2f} | Max: {np.max(cycle_times):.2f} | 95th percentile: {percentiles[2]:.2f}")
else:
    print("\nNo completed passengers to report times.")

print("\n🕒 Average Waiting Times (min):")
print(f"  Gate - regular: {waiting_gate_reg:.2f}")
print(f"  Gate - Digi:    {waiting_gate_digi:.2f}")
print(f"  Check-in:       {waiting_checkin:.2f}")
print(f"  Security (domestic): {waiting_sec_dom:.2f}")
print(f"  Security (int - first): {waiting_sec_int_first:.2f}")
print(f"  Security (int - econ):  {waiting_sec_int_econ:.2f}")
print(f"  Waiting hall before boarding: avg={waiting_hall_avg:.2f}, max={waiting_hall_max:.2f}, 95th={waiting_hall_95:.2f}")
print(f"  Boarding queue wait (avg): {boarding_queue_avg:.2f}")

print("\n📈 Average Queue Lengths:")
for k, v in avg_queue_len.items():
    print(f"  {k}: {v:.2f}")

print("\n📊 Max Queue Lengths:")
for k, v in max_queue_len.items():
    print(f"  {k}: {int(v)}")

print("\n⚙️ Resource Utilization (fraction busy):")
for k, v in utilization.items():
    print(f"  {k}: {v:.3f}")

print("\n🚪 Throughput per Hour:")
for hr, count in throughput_lines:
    print(f"{hr:02d}:00 - {count} passengers")

print("\n✈️ Breakdown by passenger type:")
for k, v in breakdown.items():
    label = "domestic" if k == "domestic" else ("int_first" if k == "int_first" else "int_econ")
    print(f"  {label}: count={v['count']}, avg total time={v['avg']:.2f} min, 95th={v['95th']:.2f} min")

print(f"\n👥 Average passengers in system (sampled trend mean): {avg_system:.2f}")
print(f"👥 Max passengers in system (sampled trend max): {max_system:.2f}")

print("\n--- End of report ---")
# ---------- SIMAN-style formatted report ----------
report_lines = []
report_lines.append("\n" + "="*60)
report_lines.append("            SIMULATION REPORT – TERMINAL 3")
report_lines.append("="*60)
report_lines.append(f"Total Runtime (sec): {round(time.time() - start_time, 2)}")
report_lines.append(f"Total Passengers Entered : {total_entered}")
report_lines.append(f"Total Passengers Completed : {len(cycle_times)}")
report_lines.append(f"Passengers Still in System : {total_entered - len(cycle_times)}")
report_lines.append("-"*60)

# Section 1: Time statistics
report_lines.append("TIME IN SYSTEM (min)")
report_lines.append(f"  Average : {np.mean(cycle_times):.2f}")
report_lines.append(f"  Minimum : {np.min(cycle_times):.2f}")
report_lines.append(f"  Maximum : {np.max(cycle_times):.2f}")
report_lines.append(f"  95th Percentile : {percentiles[2]:.2f}")
report_lines.append("-"*60)

# Section 2: Waiting times
report_lines.append("AVERAGE WAITING TIMES (minutes)")
for name, val in {
    "Gate – Regular": waiting_gate_reg,
    "Gate – DigiYatra": waiting_gate_digi,
    "Check-in": waiting_checkin,
    "Security – Domestic": waiting_sec_dom,
    "Security – Int (First)": waiting_sec_int_first,
    "Security – Int (Econ)": waiting_sec_int_econ,
    "Boarding Queue": boarding_queue_avg,
}.items():
    report_lines.append(f"  {name:<30} {val:>8.2f}")
report_lines.append("-"*60)

# Section 3: Queue statistics
report_lines.append("AVERAGE QUEUE LENGTHS")
for k, v in avg_queue_len.items():
    report_lines.append(f"  {k:<30} {v:>8.2f}")
report_lines.append("-"*60)
report_lines.append("MAXIMUM QUEUE LENGTHS")
for k, v in max_queue_len.items():
    report_lines.append(f"  {k:<30} {v:>8.2f}")
report_lines.append("-"*60)

# Section 4: Utilization
report_lines.append("RESOURCE UTILIZATION (fraction busy)")
for k, v in utilization.items():
    report_lines.append(f"  {k:<30} {v:>8.3f}")
report_lines.append("-"*60)

# Section 5: Passenger type breakdown
report_lines.append("PASSENGER TYPE SUMMARY")
for k, v in breakdown.items():
    report_lines.append(f"  {k:<15} count={v['count']:>6}, avg={v['avg']:.2f} min, 95th={v['95th']:.2f} min")
report_lines.append("-"*60)

# Section 6: System level
report_lines.append(f"Average Passengers in System : {avg_system:.2f}")
report_lines.append(f"Maximum Passengers in System : {max_system:.2f}")
report_lines.append("="*60)
report_lines.append("END OF SIMULATION REPORT")
report_lines.append("="*60)

# Print to console
print("\n".join(report_lines))

# Save to text or PDF (optional)
with open("Siman_Report_Terminal3.txt", "w") as f:
    f.write("\n".join(report_lines))

import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))
plt.plot(arrival_rate, marker='o')
plt.title("Passenger Arrival Rate per Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Passengers Arriving")
plt.grid(True)
plt.show()

"""### Basic Distributions (Cycle Times, Waits, Queues)"""

import matplotlib.pyplot as plt
import seaborn as sns

# --- 1A. Distribution of total passenger times in system ---
plt.figure(figsize=(8,5))
sns.histplot(cycle_times, bins=40, kde=True, color='skyblue')
plt.title("Distribution of Total Passenger Time in System (min)")
plt.xlabel("Total Time (min)")
plt.ylabel("Number of Passengers")
plt.show()

# --- 1B. Boxplots of waiting times across stages ---
wait_data = {
    'Gate (Regular)': waits["gate_reg"],
    'Gate (Digi)': waits["gate_digi"],
    'Check-in': waits["checkin"],
    'Security (Domestic)': waits["security_domestic"],
    'Security (Int First)': waits["security_int_first"],
    'Security (Int Econ)': waits["security_int_econ"],
    'Boarding': waits["boarding"]
}
plt.figure(figsize=(10,6))
sns.boxplot(data=list(wait_data.values()), orient='v')
plt.xticks(range(len(wait_data)), list(wait_data.keys()), rotation=45)
plt.title("Boxplot of Waiting Times at Each Stage")
plt.ylabel("Waiting Time (min)")
plt.show()

# --- 1C. Queue length distributions ---
plt.figure(figsize=(8,5))
sns.boxplot(data=list(queue_lengths.values()))
plt.xticks(range(len(queue_lengths)), list(queue_lengths.keys()), rotation=45)
plt.title("Queue Length Distributions at Different Processes")
plt.ylabel("Queue Length")
plt.show()

"""### Throughput and Passenger Flow Over Time"""

throughput_df = pd.DataFrame(throughput_lines, columns=["Hour", "Passengers"])
plt.figure(figsize=(10,5))
sns.lineplot(data=throughput_df, x="Hour", y="Passengers", marker="o")
plt.title("Passenger Throughput per Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Passengers Served")
plt.grid(True)
plt.show()

"""### Utilization and Capacity Stress"""

util_df = pd.DataFrame(list(utilization.items()), columns=["Resource", "Utilization"])
plt.figure(figsize=(8,5))
sns.barplot(data=util_df, x="Resource", y="Utilization", palette="viridis")
plt.title("Resource Utilization")
plt.ylabel("Fraction of Time Busy")
plt.xticks(rotation=45)
plt.ylim(0, max(util_df["Utilization"])*1.1)
plt.show()

"""### System Congestion (Passengers in System Over Time)"""

system_df = pd.DataFrame(sorted(system_count, key=lambda x: x[0]), columns=["Time", "Change"])
system_df["Passengers_in_System"] = system_df["Change"].cumsum()

plt.figure(figsize=(10,5))
plt.plot(system_df["Time"], system_df["Passengers_in_System"], color='orange')
plt.title("Passengers in System Over Time")
plt.xlabel("Simulation Time (minutes)")
plt.ylabel("Number of Passengers in System")
plt.grid(True)
plt.show()

"""### Passenger Type Comparison"""

types = list(breakdown.keys())
avg_times = [breakdown[t]["avg"] for t in types]
pct95_times = [breakdown[t]["95th"] for t in types]
counts = [breakdown[t]["count"] for t in types]

fig, ax1 = plt.subplots(figsize=(8,5))
sns.barplot(x=types, y=avg_times, palette="pastel", ax=ax1)
sns.pointplot(x=types, y=pct95_times, color='red', linestyles="--", ax=ax1)
plt.title("Avg vs 95th Percentile Total Time by Passenger Type")
plt.ylabel("Time (min)")
plt.xlabel("Passenger Type")
plt.legend(["95th Percentile", "Average"], loc="upper left")
plt.show()

# passenger count per type
plt.figure(figsize=(7,4))
sns.barplot(x=types, y=counts, palette="crest")
plt.title("Passenger Count per Type")
plt.ylabel("Number of Passengers")
plt.show()

"""### Queue & Wait Correlation Analysis"""

# Build dataframe for correlation
queue_wait_data = pd.DataFrame({
    "Check-in Queue": queue_lengths["checkin"],
    "Security Queue": queue_lengths["security_domestic"] + queue_lengths["security_international"],
})
queue_wait_data["Avg Wait Check-in"] = np.mean(waits["checkin"]) if waits["checkin"] else 0
queue_wait_data["Avg Wait Security"] = np.mean(waits["security_domestic"] + waits["security_int_econ"]) if (waits["security_domestic"] or waits["security_int_econ"]) else 0

sns.heatmap(queue_wait_data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Between Queue Lengths and Waiting Times")
plt.show()

"""### Dashboard-style Summary"""

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(cycle_times, bins=40, kde=True, color='skyblue', ax=axes[0,0])
axes[0,0].set_title("Distribution of Total Time in System")

sns.lineplot(x="Hour", y="Passengers", data=throughput_df, marker="o", ax=axes[0,1])
axes[0,1].set_title("Throughput per Hour")

sns.barplot(x="Resource", y="Utilization", data=util_df, palette="viridis", ax=axes[1,0])
axes[1,0].set_title("Resource Utilization")
axes[1,0].set_xticklabels(axes[1,0].get_xticklabels(), rotation=45)

axes[1,1].plot(system_df["Time"], system_df["Passengers_in_System"], color='orange')
axes[1,1].set_title("System Congestion Over Time")

plt.tight_layout()
plt.show()



import simpy
import numpy as np
import pandas as pd
import time
from statistics import mean

# ------------------- PARAMETERS -------------------
RANDOM_SEED = 42
RUN_TIME = 1440  # minutes (24 hours)
P_DIGI = 0.19
P_DOMESTIC = 0.21        # probability passenger is domestic at security

CAP_GATE_REG = 50
CAP_GATE_DIGI = 10
CAP_CHECKIN = 50
CAP_SECURITY = 70
CAP_BOARDING = 500
NUM_GATES = 62

np.random.seed(RANDOM_SEED)
start_time = time.time()

# ---------- Terminal3-specific probabilities ----------
P_INT_FIRST = 0.10

# ------------------- TIME INTERVALS -------------------
time_blocks = np.array([
    0, 60, 120, 180, 240, 300, 315, 330, 345, 360, 375, 390, 405,
    420, 435, 450, 465, 480, 495, 510, 525, 540, 600, 660, 720, 780,
    840, 900, 960, 975, 990, 1005, 1020, 1035, 1050, 1065, 1080,
    1095, 1110, 1125, 1140, 1155, 1170, 1185, 1200, 1260, 1320, 1380
])

# ------------------- SERVICE TIME ARRAYS -------------------
service_gate = np.array([
    0.6313398271, 0.6687787477, 0.6572911588, 0.6669968642, 0.675445933, 0.6894550604,
    0.6669968642, 0.6529900573, 0.6678939253, 0.661342195, 0.5564363681, 0.5408726188,
    0.5667601856, 0.5582331446, 0.5615808828, 0.5459440065, 0.5607728805, 0.5525648611,
    0.5535697277, 0.5706285908, 0.5607728805, 0.5780896302, 0.5351558127, 0.5381025232,
    0.619551916, 0.629668227, 0.6638468631, 0.6350449612, 0.6287250867, 0.6538848841,
    0.6399820735, 0.6333044569, 0.629668227, 0.6315129914, 0.6305974233, 0.6430557696,
    0.629668227, 0.6254049224, 0.6379391479, 0.6562311305, 0.6553887898, 0.6536523263,
    0.6169922097, 0.6313398271, 0.6238171069, 0.6340802719, 0.6254049224, 0.6379391479
]
)

service_checkin = np.array([
    2.415264806, 2.710918307, 2.759280499, 2.690217507, 2.684573268, 2.69566418,
    2.710918307, 2.724725597, 2.745044548, 2.724725597, 2.142850389, 2.147481033,
    2.171740491, 2.204813156, 2.19413588, 2.18199644, 2.202266172, 2.196931411,
    2.188267191, 2.168070014, 2.160307638, 2.132986495, 2.151926982, 2.147481033,
    2.466513542, 2.447419847, 2.483835843, 2.507009228, 2.396351798, 2.437119776,
    2.447419847, 2.457204988, 2.514080824, 2.527365332, 2.483835843, 2.550954592,
    2.533612721, 2.516081133, 2.496551183, 2.520546397, 2.524861739, 2.558042493,
    2.575724723, 2.567252783, 2.520546397, 2.491203285, 2.406294185, 2.496551183
])

service_domestic_security = np.array([
    1.722883499, 1.93063134, 1.93063134, 1.956965733, 2.002711293, 2.073943637,
    2.073943637, 1.956965733, 2.02272483, 2.058167496, 1.624965569, 1.66792557,
    1.588720997, 1.624965569, 1.66792557, 1.735951714, 1.710920455, 1.640591446,
    1.640591446, 1.701356473, 1.719832046, 1.607762512, 1.640591446, 1.719832046,
    1.993722175, 1.993722175, 1.980176757, 1.932793659, 1.980176757, 1.893992441,
    1.980176757, 1.949851224, 2.029161916, 1.893992441, 1.914245011, 1.871778783,
    1.893992441, 1.894360551, 1.894360551, 1.953212103, 1.997385199, 1.965389414,
    1.940136172, 2.03929951, 2.101184344, 1.835339077, 1.642278962, 1.722883499
])
service_international_security=np.array([
    1.20586958, 1.030187511, 1.105513459, 1.105513459, 1.105513459, 1.105513459,
    1.105513459, 1.030187511, 1.030187511, 1.030187511, 1.343263937, 1.343263937,
    1.343263937, 1.343263937, 1.343263937, 1.343263937, 1.343263937, 1.241884224,
    1.343263937, 1.343263937, 1.343263937, 1.343263937, 1.241884224, 1.343263937,
    1.464934502, 1.464934502, 1.464934502, 1.536452407, 1.595776007, 1.595776007,
    1.464934502, 1.595776007, 1.464934502, 1.464934502, 1.464934502, 1.464934502,
    1.595776007, 1.20586958, 1.119774699, 1.26546552, 1.20586958, 1.20586958,
    1.20586958, 1.26546552, 1.119774699, 1.119774699, 1.119774699, 1.20586958
])

# ---------- Helper ----------
def get_service_time(service_array, t):
    idx = np.searchsorted(time_blocks, t, side="right") - 1
    idx = min(max(idx, 0), len(service_array) - 1)
    return np.random.exponential(service_array[idx])

def D_gate_Digi(): return 0.5 + np.random.exponential(0.05)
def D_gate_Reg(t): return get_service_time(service_gate, t)
def D_checkin(t):  return get_service_time(service_checkin, t)
def D_security_domestic(t): return get_service_time(service_domestic_security, t)
def D_security_int_first(): return get_service_time(service_international_security, t)
def D_security_int_econ():  return get_service_time(service_domestic_security, t)
def D_walking():      return np.random.triangular(5.0, 10.0, 10.0)
def D_waiting():      return np.random.uniform(20.0, 40.0)
def D_boarding():     return np.random.triangular(5.0, 8.0, 10.0)

# ------------------- DATA -------------------
df = pd.read_csv('data/Terminal3 Flight Data.csv', skipinitialspace=True)
# Clean up: remove empty columns and ensure proper data types
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns = df.columns.str.strip()
df['Dep_time_min'] = pd.to_numeric(df['Dep_time_min'], errors='coerce').fillna(0)
df['N_passengers'] = pd.to_numeric(df['N_passengers'], errors='coerce').fillna(0).astype(int)
df['Airline'] = df['Airline'].astype(str).str.strip()
if 'Type' in df.columns:
    df['Type'] = df['Type'].astype(str).str.strip()

# ---------- Initialize Metrics ----------
cycle_times = []
total_entered = 0     # ✅ added
waits = {
    "gate_reg": [], "gate_digi": [],
    "checkin": [],
    "security_domestic": [], "security_int_first": [], "security_int_econ": [],
    "waiting_hall": [], "boarding": []
}
queue_lengths = {
    "checkin": [], "security_domestic": [], "security_international": [], "boarding_domestic": [], "boarding_international": []
}
busy_time = {"checkin": 0.0, "security_domestic": 0.0, "security_int_first": 0.0, "security_int_econ": 0.0, "boarding_domestic": 0.0, "boarding_international": 0.0}
throughput = np.zeros(24)
results_by_type = {"domestic": [], "int_first": [], "int_econ": []}
system_count = []

num_gates_domestic = max(1, int(round(NUM_GATES * P_DOMESTIC)))
num_gates_international = NUM_GATES - num_gates_domestic

# ---------- Passenger Process ----------
def passenger(env, pid, dep_time, ptype, gate_reg, gate_digi, checkins, securities_dom, securities_int, boardings_dom, boardings_int):
    t0 = env.now
    system_count.append((env.now, +1))

    if np.random.rand() < P_DIGI:
        chosen = min(gate_digi, key=lambda r: len(r.queue))
        with chosen.request() as req:
            t_req = env.now; yield req
            waits["gate_digi"].append(env.now - t_req)
            yield env.timeout(D_gate_Digi())
    else:
        chosen = min(gate_reg, key=lambda r: len(r.queue))
        with chosen.request() as req:
            t_req = env.now; yield req
            waits["gate_reg"].append(env.now - t_req)
            yield env.timeout(D_gate_Reg(env.now))

    chosen = min(checkins, key=lambda r: len(r.queue))
    queue_lengths["checkin"].append(len(chosen.queue))
    with chosen.request() as req:
        t_req = env.now; yield req
        waits["checkin"].append(env.now - t_req)
        st = D_checkin(env.now)
        yield env.timeout(st)
        busy_time["checkin"] += st

    if ptype == "D":
        chosen = min(securities_dom, key=lambda r: len(r.queue))
        queue_lengths["security_domestic"].append(len(chosen.queue))
        with chosen.request() as req:
            t_req = env.now; yield req
            waits["security_domestic"].append(env.now - t_req)
            st = D_security_domestic(env.now)
            yield env.timeout(st)
            busy_time["security_domestic"] += st

        chosen_board = min(boardings_dom, key=lambda r: len(r.queue))
        queue_lengths["boarding_domestic"].append(len(chosen_board.queue))
        with chosen_board.request() as req:
            t_req = env.now; yield req
            waits["boarding"].append(env.now - t_req)
            st2 = D_boarding()
            yield env.timeout(st2)
            busy_time["boarding_domestic"] += st2
        results_by_type["domestic"].append(env.now - t0)

    elif ptype == "I":
        if np.random.rand() < 0.2:
            sec_type = "security_int_first"
            chosen = min(securities_int, key=lambda r: len(r.queue))
            queue_lengths["security_international"].append(len(chosen.queue))
            with chosen.request() as req:
                t_req = env.now; yield req
                waits["security_int_first"].append(env.now - t_req)
                st = D_security_int_first()
                yield env.timeout(st)
                busy_time["security_int_first"] += st

            chosen_board = min(boardings_int, key=lambda r: len(r.queue))
            queue_lengths["boarding_international"].append(len(chosen_board.queue))
            with chosen_board.request() as req:
                t_req = env.now; yield req
                waits["boarding"].append(env.now - t_req)
                st2 = D_boarding()
                yield env.timeout(st2)
                busy_time["boarding_international"] += st2
            results_by_type["int_first"].append(env.now - t0)
        else:
            sec_type = "security_int_econ"
            chosen = min(securities_int, key=lambda r: len(r.queue))
            queue_lengths["security_international"].append(len(chosen.queue))
            with chosen.request() as req:
                t_req = env.now; yield req
                waits["security_int_econ"].append(env.now - t_req)
                st = D_security_int_econ()
                yield env.timeout(st)
                busy_time["security_int_econ"] += st

            chosen_board = min(boardings_int, key=lambda r: len(r.queue))
            queue_lengths["boarding_international"].append(len(chosen_board.queue))
            with chosen_board.request() as req:
                t_req = env.now; yield req
                waits["boarding"].append(env.now - t_req)
                st2 = D_boarding()
                yield env.timeout(st2)
                busy_time["boarding_international"] += st2
            results_by_type["int_econ"].append(env.now - t0)

    waiting_time = D_waiting()
    waits["waiting_hall"].append(waiting_time)
    yield env.timeout(D_walking() + waiting_time)

    cycle_times.append(env.now - t0)
    idx = min(int(env.now // 60), len(throughput) - 1)
    throughput[idx] += 1
    system_count.append((env.now, -1))

# ---------- Flight Source ----------
def flight_source(env, gate_reg, gate_digi, checkin_list,
                  securities_dom, securities_int, boardings_dom, boardings_int):
    global total_entered  # ✅ added
    pid = 0
    for _, row in df.iterrows():
        dep_time = float(row["Dep_time_min"])
        n_passengers = int(row["N_passengers"])
        arr_start = dep_time - 120
        arr_end = dep_time - 60

        if arr_start < 0:
            arrival_times = np.random.uniform(0, max(arr_end, dep_time, 0), n_passengers)
        else:
            arrival_times = np.random.uniform(arr_start, arr_end, n_passengers)

        arrival_times = np.clip(arrival_times, 0, RUN_TIME)
        arrival_times.sort()

        for arr_time in arrival_times:
            yield env.timeout(max(0, arr_time - env.now))
            pid += 1
            total_entered += 1   # ✅ increment total entered
            ptype = str(row["Type"]).strip().upper()
            env.process(
                passenger(
                    env, pid, dep_time, ptype,
                    gate_reg, gate_digi,
                    checkin_list, securities_dom, securities_int,
                    boardings_dom, boardings_int
                )
            )

# ---------- Simulation set-up ----------
env = simpy.Environment()
gate_reg = [simpy.Resource(env, capacity=1) for _ in range(CAP_GATE_REG)]
gate_digi = [simpy.Resource(env, capacity=1) for _ in range(CAP_GATE_DIGI)]
checkins = [simpy.Resource(env, capacity=1) for _ in range(CAP_CHECKIN)]
sec_dom_count = max(1, int(round(CAP_SECURITY * P_DOMESTIC)))
sec_int_count = max(1, CAP_SECURITY - sec_dom_count)
securities_dom = [simpy.Resource(env, capacity=1) for _ in range(sec_dom_count)]
securities_int = [simpy.Resource(env, capacity=1) for _ in range(sec_int_count)]
boardings_dom = [simpy.Resource(env, capacity=CAP_BOARDING) for _ in range(num_gates_domestic)]
boardings_int = [simpy.Resource(env, capacity=CAP_BOARDING) for _ in range(num_gates_international)]
env.process(flight_source(env, gate_reg, gate_digi, checkins, securities_dom, securities_int, boardings_dom, boardings_int))

print("\n--- Simulation Started (Terminal 3) ---\n")
env.run(until=RUN_TIME+2000)
print("\n--- Simulation Completed ---")
print(f"Total runtime: {round(time.time() - start_time, 2)} seconds")

# ---------- Metrics ----------
def avg(lst): return mean(lst) if lst else 0.0
def safe_max(lst): return np.max(lst) if len(lst) else 0.0

avg_queue_len = {k: np.mean(v) if len(v) else 0.0 for k, v in queue_lengths.items()}
max_queue_len = {k: safe_max(v) for k, v in queue_lengths.items()}

utilization = {}
utilization["checkin"] = busy_time["checkin"] / (len(checkins) * RUN_TIME) if len(checkins) else 0.0
utilization["security_domestic"] = busy_time["security_domestic"] / (len(securities_dom) * RUN_TIME) if len(securities_dom) else 0.0
utilization["security_int_first"] = busy_time["security_int_first"] / (len(securities_int) * RUN_TIME) if len(securities_int) else 0.0
utilization["security_int_econ"] = busy_time["security_int_econ"] / (len(securities_int) * RUN_TIME) if len(securities_int) else 0.0
utilization["boarding_domestic"] = busy_time["boarding_domestic"] / (len(boardings_dom) * RUN_TIME) if len(boardings_dom) else 0.0
utilization["boarding_international"] = busy_time["boarding_international"] / (len(boardings_int) * RUN_TIME) if len(boardings_int) else 0.0

percentiles = np.percentile(cycle_times, [5, 50, 95]) if cycle_times else [0, 0, 0]
system_trend = np.cumsum([c for _, c in sorted(system_count, key=lambda x: x[0])]) if system_count else np.array([0])
avg_system = np.mean(system_trend) if len(system_trend) else 0.0
max_system = np.max(system_trend) if len(system_trend) else 0.0

# ---------- Print results ----------
print(f"\nTotal passengers entered: {total_entered}")
print(f"Total passengers completed: {len(cycle_times)}")
print(f"Passengers still in system (not completed): {total_entered - len(cycle_times)}")

print(f"\nAverage total time in system: {np.mean(cycle_times):.2f} min" if cycle_times else "No completed passengers")
print(f"Min: {np.min(cycle_times):.2f} | Max: {np.max(cycle_times):.2f} | 95th percentile: {percentiles[2]:.2f}" if cycle_times else "")

print("\n--- End of report ---")

