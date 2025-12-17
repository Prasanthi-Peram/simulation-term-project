

"""Terminal 1 Code

"""

import simpy
import numpy as np
import pandas as pd
import time
from statistics import mean

# ------------------- PARAMETERS -------------------
RANDOM_SEED = 42
RUN_TIME = 1440  # minutes (24 hours)
P_DIGI = 0.19

CAP_GATE_REG = 24
CAP_GATE_DIGI = 6
CAP_CHECKIN_INDIGO = 20
CAP_CHECKIN_SPICEJET = 10
CAP_SECURITY = 42
CAP_BOARDING = 50
NUM_GATES = 23

np.random.seed(RANDOM_SEED)
start_time = time.time()

# ------------------- TIME INTERVALS -------------------
time_blocks = np.array([
    0, 60, 120, 180, 240, 300, 315, 330, 345, 360, 375, 390, 405,
    420, 435, 450, 465, 480, 495, 510, 525, 540, 600, 660, 720, 780,
    840, 900, 960, 975, 990, 1005, 1020, 1035, 1050, 1065, 1080,
    1095, 1110, 1125, 1140, 1155, 1170, 1185, 1200, 1260, 1320, 1380
])

# ------------------- REORDERED SERVICE ARRAYS -------------------
service_gate = np.array([
1.873562059,1.873562059,2.607665649,2.607665649,2.694703497,3.207665649,3.273359262,
2.875598168,3.004047443,3.757170457,4.039022931,1.607947608,1.665027568,1.607947608,
1.607947608,1.637550977,1.637550977,1.643207992,1.625980035,1.643207992,1.654275554,
1.614051694,1.595447002,1.620061696,1.548325568,1.577875003,1.595044445,1.540989546,
1.577875003,1.589409089,1.547390396,1.577875003,1.540989546,1.540989546,1.534480629,
1.56597408,1.583686703,1.589409089,1.900987697,1.909843151,1.882851225,1.918561903,
1.94393393,1.900987697,1.882851225,1.882851225,1.873562059,1.882851225,1.918561903
])

service_checkin = np.array([
2.091022075,2.777475291,2.721808294,4.037687461,4.361985378,5.659198478,5.60159215,
5.321820993,5.235947657,1.876065026,1.91119625,1.957274516,1.968305721,1.96483997,
1.948725962,1.927793286,1.936338341,1.936338341,1.948725962,1.918230719,1.881203234,
1.955236711,1.890764707,1.824491999,1.802232744,1.76626219,1.515708519,1.64136654,
1.659183649,1.64136654,1.781972235,1.675375371,1.774365785,1.789126533,1.64136654,
1.781972235,2.340199125,2.360055764,2.461985615,2.422508946,2.174891424,2.232827924,
2.038678385,2.038678385,2.174891424,2.002377287,1.802791849,2.195701591
])

service_security = np.array([
4.01075125,4.877720533,4.365247184,4.285726669,5.01075125,5.877720533,8.01075125,
9.285726669,9.12013644,3.701277027,3.701277027,3.827918394,3.701277027,3.788309462,
3.701277027,3.601809427,3.653273102,3.601809427,3.601809427,3.653273102,3.701277027,
3.653273102,3.546457181,3.567928666,3.51829196,3.567928666,3.51829196,3.567928666,
3.65747274,3.51829196,3.464874869,3.464874869,3.614203796,3.51829196,3.51829196,
3.698039578,4.539139865,4.473779257,4.40402282,4.539139865,4.329348649,4.249138637,
4.329348649,4.329348649,4.329348649,4.473779257,4.539139865,4.658390106
])

# ---------- Helper ----------
def get_service_time(service_array, t):
    idx = np.searchsorted(time_blocks, t, side="right") - 1
    idx = min(max(idx, 0), len(service_array) - 1)
    return np.random.exponential(service_array[idx])

def D_gate_Digi_T1(): return 0.5 + np.random.exponential(0.05)
def D_gate_Reg_T1(t): return get_service_time(service_gate, t)
def D_checkin_T1(t):  return get_service_time(service_checkin, t)
def D_security_T1(t): return get_service_time(service_security, t)
def D_walking():      return np.random.triangular(5.0, 10.0, 10.0)
def D_waiting():      return np.random.uniform(20.0, 40.0)
def D_boarding():     return np.random.triangular(5.0, 8.0, 10.0)

# ---------- Data ----------
df = pd.read_csv('data/Terminal1 Flight Data.csv')

# ---------- Initialize Metrics ----------
cycle_times = []
waits = {"gate_reg": [], "gate_digi": [], "checkin_indigo": [], "checkin_spicejet": [],
         "security": [], "boarding": [], "waiting_hall": []}
queue_lengths = {"checkin": [], "security": [], "boarding": []}
busy_time = {"checkin": 0, "security": 0, "boarding": 0}
throughput = np.zeros(28)  # for 27 hours
results_by_airline = {"IndiGo": [], "SpiceJet": []}
system_count = []

# ---------- Passenger Process ----------
def passenger(env, pid, airline, gate_reg, gate_digi, checkin_indigo, checkin_spicejet, securities, boardings):
    t0 = env.now
    system_count.append((env.now, +1))

    # Gate (Digital or Regular)
    if np.random.rand() < P_DIGI:
        chosen = min(gate_digi, key=lambda r: len(r.queue))
        with chosen.request() as req:
            t_req = env.now; yield req
            waits["gate_digi"].append(env.now - t_req)
            yield env.timeout(D_gate_Digi_T1())
    else:
        chosen = min(gate_reg, key=lambda r: len(r.queue))
        with chosen.request() as req:
            t_req = env.now; yield req
            waits["gate_reg"].append(env.now - t_req)
            yield env.timeout(D_gate_Reg_T1(env.now))

    # Check-in
    if airline.lower() == "indigo":
        chosen = min(checkin_indigo, key=lambda r: len(r.queue))
        queue_lengths["checkin"].append(len(chosen.queue))
        with chosen.request() as req:
            t_req = env.now; yield req
            waits["checkin_indigo"].append(env.now - t_req)
            st = D_checkin_T1(env.now)
            yield env.timeout(st)
            busy_time["checkin"] += st
    else:
        chosen = min(checkin_spicejet, key=lambda r: len(r.queue))
        queue_lengths["checkin"].append(len(chosen.queue))
        with chosen.request() as req:
            t_req = env.now; yield req
            waits["checkin_spicejet"].append(env.now - t_req)
            st = D_checkin_T1(env.now)
            yield env.timeout(st)
            busy_time["checkin"] += st

    # Security
    chosen = min(securities, key=lambda r: len(r.queue))
    queue_lengths["security"].append(len(chosen.queue))
    with chosen.request() as req:
        t_req = env.now; yield req
        waits["security"].append(env.now - t_req)
        st = D_security_T1(env.now)
        yield env.timeout(st)
        busy_time["security"] += st

    # Waiting Hall (before boarding)
    waiting_time = D_waiting()
    waits["waiting_hall"].append(waiting_time)
    yield env.timeout(D_walking() + waiting_time)

    # Boarding
    chosen = min(boardings, key=lambda r: len(r.queue))
    queue_lengths["boarding"].append(len(chosen.queue))
    with chosen.request() as req:
        t_req = env.now; yield req
        waits["boarding"].append(env.now - t_req)
        st = D_boarding()
        yield env.timeout(st)
        busy_time["boarding"] += st

    cycle_times.append(env.now - t0)
    idx = min(int(env.now // 60), len(throughput) - 1)
    throughput[idx] += 1
    results_by_airline[airline].append(env.now - t0)
    system_count.append((env.now, -1))

# ---------- Flight Source ----------
def flight_source(env, gate_reg, gate_digi, checkin_indigo, checkin_spicejet, securities, boardings):
    pid = 0
    eps = 1e-6
    for _, row in df.iterrows():
        dep_time = row["Dep_time_min"]
        n_passengers = int(row["N_passengers"])
        airline = row["Airline"]
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
        mean_arrival_gap = total_window / n_passengers
        inter_arrivals = np.random.exponential(scale=mean_arrival_gap, size=n_passengers)

        # normalize so arrivals fit inside window
        arrival_times = arr_start + np.cumsum(inter_arrivals)
        arrival_times = np.clip(arrival_times, arr_start, arr_end)
        arrival_times.sort()
        for arr_time in arrival_times:
            yield env.timeout(max(0, arr_time - env.now))
            pid += 1
            env.process(passenger(env, pid, airline, gate_reg, gate_digi, checkin_indigo, checkin_spicejet, securities, boardings))

# ---------- Simulation ----------
env = simpy.Environment()
gate_reg = [simpy.Resource(env, capacity=1) for _ in range(CAP_GATE_REG)]
gate_digi = [simpy.Resource(env, capacity=1) for _ in range(CAP_GATE_DIGI)]
checkin_indigo = [simpy.Resource(env, capacity=1) for _ in range(CAP_CHECKIN_INDIGO)]
checkin_spicejet = [simpy.Resource(env, capacity=1) for _ in range(CAP_CHECKIN_SPICEJET)]
securities = [simpy.Resource(env, capacity=1) for _ in range(CAP_SECURITY)]
boardings = [simpy.Resource(env, capacity=CAP_BOARDING) for _ in range(NUM_GATES)]

env.process(flight_source(env, gate_reg, gate_digi, checkin_indigo, checkin_spicejet, securities, boardings))

print("\n--- Simulation Started ---\n")
env.run(until=RUN_TIME + 180)

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
def safe_max(lst): return np.max(lst) if len(lst) > 0 else 0

avg_queue_len = {k: np.mean(v) if len(v) else 0 for k, v in queue_lengths.items()}
max_queue_len = {k: safe_max(v) for k, v in queue_lengths.items()}
utilization = {k: busy_time[k] / RUN_TIME for k in busy_time}
percentiles = np.percentile(cycle_times, [5, 50, 95])
system_trend = np.cumsum([c for _, c in sorted(system_count, key=lambda x: x[0])])
avg_system = np.mean(system_trend)
max_system = np.max(system_trend)

# ---------- Results ----------
print(f"\nTotal passengers completed: {len(cycle_times)}")
print(f"Average total time in system: {np.mean(cycle_times):.2f} min")
print(f"Min: {np.min(cycle_times):.2f} | Max: {np.max(cycle_times):.2f} | 95th percentile: {percentiles[2]:.2f}")

print("\n🕒 Average Waiting Times (min):")
for k, v in waits.items():
    print(f"  {k}: {avg(v):.2f}")
if "waiting_hall" in waits:
    print(f"  ➤ Waiting Hall (before boarding): avg={avg(waits['waiting_hall']):.2f}, "
          f"max={safe_max(waits['waiting_hall']):.2f}, 95th={np.percentile(waits['waiting_hall'],95):.2f}")

print("\n📈 Average Queue Lengths:", avg_queue_len)
print("📊 Max Queue Lengths:", max_queue_len)
print("⚙️ Resource Utilization:", utilization)

print("\n🚪 Throughput per Hour:")
for h in range(len(throughput)):
    print(f"{h:02d}:00 - {throughput[h]} passengers")

print("\n✈️ Breakdown by Airline:")
for a in results_by_airline:
    if len(results_by_airline[a]) > 0:
        print(f"  {a}: avg={np.mean(results_by_airline[a]):.2f}, count={len(results_by_airline[a])}")

print(f"\n👥 Average passengers in system: {avg_system:.2f}")
print(f"👥 Max passengers in system: {max_system:.2f}")
# ---------- SIMAN-style formatted report (Terminal 1) ----------
report_lines = []
report_lines.append("\n" + "="*60)
report_lines.append("            SIMULATION REPORT – TERMINAL 1")
report_lines.append("="*60)
report_lines.append(f"Total Runtime (sec): {round(time.time() - start_time, 2)}")
report_lines.append(f"Total Passengers Entered : {len(system_count)/2}") # Assuming +1 and -1 for each passenger
report_lines.append(f"Total Passengers Completed : {len(cycle_times)}")
report_lines.append(f"Passengers Still in System : {len(system_count)/2 - len(cycle_times)}") # Assuming +1 and -1 for each passenger
report_lines.append("-"*60)

# Section 1: Time statistics
report_lines.append("TIME IN SYSTEM (min)")
report_lines.append(f"  Average : {np.mean(cycle_times):.2f}")
report_lines.append(f"  Minimum : {np.min(cycle_times):.2f}")
report_lines.append(f"  Maximum : {np.max(cycle_times):.2f}")
report_lines.append(f"  95th Percentile : {np.percentile(cycle_times, 95):.2f}")
report_lines.append("-"*60)

# Section 2: Waiting times
report_lines.append("AVERAGE WAITING TIMES (minutes)")
report_lines.append(f"  {'Gate – Regular':<30} {avg(waits['gate_reg']):>8.2f}")
report_lines.append(f"  {'Gate – DigiYatra':<30} {avg(waits['gate_digi']):>8.2f}")
report_lines.append(f"  {'Check-in (IndiGo)':<30} {avg(waits['checkin_indigo']):>8.2f}")
report_lines.append(f"  {'Check-in (SpiceJet)':<30} {avg(waits['checkin_spicejet']):>8.2f}")
report_lines.append(f"  {'Security':<30} {avg(waits['security']):>8.2f}")
report_lines.append(f"  {'Boarding Queue':<30} {avg(waits['boarding']):>8.2f}")
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

# Section 5: Passenger statistics
report_lines.append("PASSENGER SUMMARY")
for k, v in results_by_airline.items():
    if len(v) > 0:
        report_lines.append(f"  {k:<15} count={len(v):>6}, avg={np.mean(v):.2f} min, 95th={np.percentile(v, 95):.2f} min")
report_lines.append("-"*60)

# Section 6: System-level
report_lines.append(f"Average Passengers in System : {avg_system:.2f}")
report_lines.append(f"Maximum Passengers in System : {max_system:.2f}")
report_lines.append("="*60)
report_lines.append("END OF SIMULATION REPORT")
report_lines.append("="*60)

# Print to console
print("\n".join(report_lines))

# Save as text file
with open("Siman_Report_Terminal1.txt", "w") as f:
    f.write("\n".join(report_lines))

"""### Passenger Throughput per Hour (Terminal 1)
Shows how many passengers completed the process each hour, helping identify peak throughput times.
"""

import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))
plt.plot(arrival_rate, marker='o')
plt.title("Passenger Arrival Rate per Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Passengers Arriving")
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10,5))
plt.plot(range(len(throughput)), throughput, marker='o', linewidth=2)
plt.title("Passenger Throughput per Hour (Terminal 1)")
plt.xlabel("Hour of Day")
plt.ylabel("Passengers Completed")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

"""### Passengers in System Over Time (T1)
Tracks how the total number of passengers in the system evolves over time.
"""

times, deltas = zip(*sorted(system_count, key=lambda x: x[0]))
system_trend = np.cumsum(deltas)
plt.figure(figsize=(10,5))
plt.plot(times, system_trend, color='teal')
plt.title("Number of Passengers in System Over Time (T1)")
plt.xlabel("Simulation Time (min)")
plt.ylabel("Passengers in System")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

"""### Average Waiting Time per Process (T1)

Shows mean waiting time across each process — check-in, security, etc.
"""

avg_waits = {k: np.mean(v) if len(v) else 0 for k,v in waits.items()}
plt.figure(figsize=(8,5))
plt.bar(avg_waits.keys(), avg_waits.values(), color='cornflowerblue', alpha=0.8)
plt.title("Average Waiting Time per Process (T1)")
plt.ylabel("Average Waiting Time (min)")
plt.xticks(rotation=25)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

"""### Distribution of Waiting Times - Security (T1)

Examines how variable the waiting times are for the security process.
"""

if "security" in waits and len(waits["security"]) > 0:
    plt.figure(figsize=(8,5))
    plt.hist(waits["security"], bins=30, color='lightcoral', alpha=0.7, edgecolor='black')
    plt.title("Distribution of Waiting Times - Security (T1)")
    plt.xlabel("Waiting Time (min)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

"""### Average Queue Length per Process (T1)

Compares mean queue lengths across system processes.
"""

plt.figure(figsize=(8,5))
plt.bar(queue_lengths.keys(), [np.mean(v) if v else 0 for v in queue_lengths.values()],
        color='mediumseagreen', alpha=0.8)
plt.title("Average Queue Length per Process (T1)")
plt.ylabel("Average Queue Length")
plt.xticks(rotation=25)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

"""### Resource Utilization (T1)

Shows how busy each resource was during the simulation — a measure of bottleneck potential.
"""

if "utilization" in locals():
    plt.figure(figsize=(8,5))
    plt.bar(utilization.keys(), [v*100 for v in utilization.values()], color='goldenrod', alpha=0.9)
    plt.title("Resource Utilization (%) (T1)")
    plt.ylabel("Utilization (%)")
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

"""### Distribution of Total Passenger Cycle Times (T1)

Analyzes how long passengers spend in the system overall.
"""

plt.figure(figsize=(8,5))
plt.hist(cycle_times, bins=50, color='mediumpurple', alpha=0.75, edgecolor='black')
plt.title("Distribution of Total Passenger Cycle Times (T1)")
plt.xlabel("Total Time in System (min)")
plt.ylabel("Number of Passengers")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

"""### Average Total Time in System by Airline (T1)

If multiple airlines operate in T1, this compares their mean total times.
"""

if "results_by_airline" in locals() and len(results_by_airline) > 0:
    airlines = list(results_by_airline.keys())
    avg_times = [np.mean(results_by_airline[a]) if len(results_by_airline[a]) > 0 else 0 for a in airlines]
    plt.figure(figsize=(8,5))
    plt.bar(airlines, avg_times, color=['deepskyblue', 'orange'])
    plt.title("Average Total Time in System by Airline (T1)")
    plt.ylabel("Average Time (min)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

"""### Cumulative Passengers Served Over Time (T1)

Shows total completed passengers as simulation progresses.
"""

plt.figure(figsize=(9,5))
plt.plot(np.cumsum(throughput), color='darkorange', linewidth=2)
plt.title("Cumulative Passengers Served Over Time (T1)")
plt.xlabel("Hour")
plt.ylabel("Cumulative Passengers")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

"""### Correlation: Total Waiting Time vs Total System Time (T1)

Tests whether passengers who wait longer also spend longer in the full system.
"""

total_wait = []
for i in range(len(cycle_times)):
    total_wait_val = 0
    for k, v in waits.items():
        if i < len(v):
            total_wait_val += v[i]
    total_wait.append(total_wait_val)

plt.figure(figsize=(7,5))
plt.scatter(total_wait, cycle_times[:len(total_wait)], alpha=0.3, color='slateblue')
plt.title("Correlation: Total Waiting Time vs Total System Time (T1)")
plt.xlabel("Total Waiting Time (min)")
plt.ylabel("Total System Time (min)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

"""### Waiting Time vs Utilization per Process

Shows how waiting times scale with resource utilization — helps find bottlenecks.
"""

if "utilization" in locals():
    # Get the common keys between avg_waits and utilization
    common_keys = list(set(avg_waits.keys()) & set(utilization.keys()))

    plt.figure(figsize=(8,5))
    plt.scatter([utilization[k]*100 for k in common_keys], [avg_waits[k] for k in common_keys],
                color='royalblue', alpha=0.7)
    plt.title("Waiting Time vs Resource Utilization (T1)")
    plt.xlabel("Utilization (%)")
    plt.ylabel("Average Waiting Time (min)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

"""### Process Comparison - Queue Length vs Waiting Time

Helps identify whether delays are caused by queue build-up or inherent service times.
"""

plt.figure(figsize=(8,5))
processes = list(avg_waits.keys())
avg_q = [np.mean(queue_lengths[p]) if p in queue_lengths and len(queue_lengths[p])>0 else 0 for p in processes]
plt.scatter(avg_q, [avg_waits[p] for p in processes], s=100, color='tomato', alpha=0.8)
for i, txt in enumerate(processes):
    plt.text(avg_q[i]+0.05, avg_waits[processes[i]]+0.05, txt)
plt.title("Queue Length vs Waiting Time by Process (T1)")
plt.xlabel("Average Queue Length")
plt.ylabel("Average Waiting Time (min)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# ====================================================
# 📈 TERMINAL 1 PERFORMANCE ANALYSIS PLOTS
# ====================================================

plt.style.use('seaborn-v0_8-colorblind')

# 1️⃣ Throughput per Hour
plt.figure(figsize=(10,5))
plt.plot(range(len(throughput)), throughput, marker='o', linewidth=2)
plt.title("Passenger Throughput per Hour (Terminal 1)")
plt.xlabel("Hour of Day")
plt.ylabel("Passengers Completed")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 2️⃣ Passengers in System Over Time
times, deltas = zip(*sorted(system_count, key=lambda x: x[0]))
system_trend = np.cumsum(deltas)
plt.figure(figsize=(10,5))
plt.plot(times, system_trend, color='teal')
plt.title("Number of Passengers in System Over Time (T1)")
plt.xlabel("Simulation Time (min)")
plt.ylabel("Passengers in System")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 3️⃣ Average Waiting Time per Process
avg_waits = {k: np.mean(v) if len(v) else 0 for k,v in waits.items()}
plt.figure(figsize=(8,5))
plt.bar(avg_waits.keys(), avg_waits.values(), color='cornflowerblue', alpha=0.8)
plt.title("Average Waiting Time per Process (T1)")
plt.ylabel("Average Waiting Time (min)")
plt.xticks(rotation=25)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 4️⃣ Waiting Time Distribution - Security
plt.figure(figsize=(8,5))
if "security" in waits and len(waits["security"]) > 0:
    plt.hist(waits["security"], bins=30, color='lightcoral', alpha=0.7, edgecolor='black')
    plt.title("Distribution of Waiting Times - Security (T1)")
    plt.xlabel("Waiting Time (min)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# 5️⃣ Average Queue Length per Process
plt.figure(figsize=(8,5))
plt.bar(queue_lengths.keys(), [np.mean(v) if v else 0 for v in queue_lengths.values()],
        color='mediumseagreen', alpha=0.8)
plt.title("Average Queue Length per Process (T1)")
plt.ylabel("Average Queue Length")
plt.xticks(rotation=25)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 6️⃣ Resource Utilization
if "utilization" in locals():
    plt.figure(figsize=(8,5))
    plt.bar(utilization.keys(), [v*100 for v in utilization.values()], color='goldenrod', alpha=0.9)
    plt.title("Resource Utilization (%) (T1)")
    plt.ylabel("Utilization (%)")
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# 7️⃣ Cycle Time Distribution
plt.figure(figsize=(8,5))
plt.hist(cycle_times, bins=50, color='mediumpurple', alpha=0.75, edgecolor='black')
plt.title("Distribution of Total Passenger Cycle Times (T1)")
plt.xlabel("Total Time in System (min)")
plt.ylabel("Number of Passengers")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 8️⃣ Comparison by Airline
if "results_by_airline" in locals() and len(results_by_airline) > 0:
    airlines = list(results_by_airline.keys())
    avg_times = [np.mean(results_by_airline[a]) if len(results_by_airline[a]) > 0 else 0 for a in airlines]
    plt.figure(figsize=(8,5))
    plt.bar(airlines, avg_times, color=['deepskyblue', 'orange'])
    plt.title("Average Total Time in System by Airline (T1)")
    plt.ylabel("Average Time (min)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# 9️⃣ Cumulative Passengers Served
plt.figure(figsize=(9,5))
plt.plot(np.cumsum(throughput), color='darkorange', linewidth=2)
plt.title("Cumulative Passengers Served Over Time (T1)")
plt.xlabel("Hour")
plt.ylabel("Cumulative Passengers")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 🔟 Waiting vs System Time Correlation
total_wait = []
for i in range(len(cycle_times)):
    total_wait_val = 0
    for k, v in waits.items():
        if i < len(v):
            total_wait_val += v[i]
    total_wait.append(total_wait_val)

plt.figure(figsize=(7,5))
plt.scatter(total_wait, cycle_times[:len(total_wait)], alpha=0.3, color='slateblue')
plt.title("Correlation: Total Waiting Time vs Total System Time (T1)")
plt.xlabel("Total Waiting Time (min)")
plt.ylabel("Total System Time (min)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

"""Simon Report Dummy

"""

from datetime import datetime

# Compute statistics
avg_cycle = mean(cycle_times) if cycle_times else 0
avg_gate_reg = mean(wait_gate_reg) if wait_gate_reg else 0
avg_gate_digi = mean(wait_gate_digi) if wait_gate_digi else 0
avg_checkin = mean(wait_checkin) if wait_checkin else 0
avg_security = mean(wait_security) if wait_security else 0
avg_boarding = mean(wait_boarding) if wait_boarding else 0

# Timestamp and header info
run_date = datetime.now().strftime("%b %d, %Y")
rep_end_time = RUN_TIME
rep_end_datetime = datetime.now().strftime("%A, %B %d, %Y, %I:%M:%S %p")

print(f"""
{"ARENA SIMULATION RESULTS".center(90)}
{"SAMPATH - License: STUDENT".center(90)}

{"Summary for Replication 1 of 1".center(90)}

Project: Delhi Airport Passenger Flow Simulation{' ' * 20}Run execution date : {run_date}
Analyst: Adepu Sampath{' ' * 37}Model revision date: {run_date}

Replication ended at time      : {rep_end_time:>6.1f} Minutes ({rep_end_datetime})
Base Time Units: Minutes

{"TALLY VARIABLES".center(90, " ")}
{"Identifier":<45}{"Average":>10}{"Half Width":>12}{"Minimum":>10}{"Maximum":>10}{"Observations":>12}
{"_"*90}

{"CycleTime":<45}{avg_cycle:>10.2f}{'':>12}{min(cycle_times):>10.2f}{max(cycle_times):>10.2f}{len(cycle_times):>12}
{"Gate (Regular).Queue.WaitingTime":<45}{avg_gate_reg:>10.3f}{'':>12}{min(wait_gate_reg) if wait_gate_reg else 0:>10.3f}{max(wait_gate_reg) if wait_gate_reg else 0:>10.3f}{len(wait_gate_reg):>12}
{"Gate (DigiYatra).Queue.WaitingTime":<45}{avg_gate_digi:>10.3f}{'':>12}{min(wait_gate_digi) if wait_gate_digi else 0:>10.3f}{max(wait_gate_digi) if wait_gate_digi else 0:>10.3f}{len(wait_gate_digi):>12}
{"Checkin.Queue.WaitingTime":<45}{avg_checkin:>10.3f}{'':>12}{min(wait_checkin) if wait_checkin else 0:>10.3f}{max(wait_checkin) if wait_checkin else 0:>10.3f}{len(wait_checkin):>12}
{"Security.Queue.WaitingTime":<45}{avg_security:>10.3f}{'':>12}{min(wait_security) if wait_security else 0:>10.3f}{max(wait_security) if wait_security else 0:>10.3f}{len(wait_security):>12}
{"Boarding.Queue.WaitingTime":<45}{avg_boarding:>10.3f}{'':>12}{min(wait_boarding) if wait_boarding else 0:>10.3f}{max(wait_boarding) if wait_boarding else 0:>10.3f}{len(wait_boarding):>12}

{"DISCRETE-CHANGE VARIABLES".center(90, " ")}
{"Identifier":<45}{"Average":>10}{"Half Width":>12}{"Minimum":>10}{"Maximum":>10}{"Final Value":>12}
{"_"*90}

{"Gate (Regular).Utilization":<45}{avg_gate_reg/(avg_gate_reg+avg_cycle):>10.3f}{'':>12}{0:>10.3f}{1.0:>10.3f}{'--':>12}
{"Checkin.Utilization":<45}{avg_checkin/(avg_checkin+avg_cycle):>10.3f}{'':>12}{0:>10.3f}{1.0:>10.3f}{'--':>12}
{"Security.Utilization":<45}{avg_security/(avg_security+avg_cycle):>10.3f}{'':>12}{0:>10.3f}{1.0:>10.3f}{'--':>12}

{"OUTPUTS".center(90, " ")}
{"Identifier":<45}{"Value":>10}
{"_"*60}
{"Total.Passengers.Arrived":<45}{len(total_arrivals):>10.0f}
{"Total.Passengers.Completed":<45}{len(cycle_times):>10.0f}
{"System.AverageTimeInSystem":<45}{avg_cycle:>10.2f}
{"System.AverageCheckinWait":<45}{avg_checkin:>10.2f}
{"System.AverageSecurityWait":<45}{avg_security:>10.2f}
{"System.AverageBoardingWait":<45}{avg_boarding:>10.2f}

Simulation run time: {round(time.time() - start_time, 2)} minutes.
Simulation run complete.
""")

"""Terminal 2"""

import simpy
import numpy as np
import pandas as pd
import time
from statistics import mean

# ------------------- PARAMETERS -------------------
RANDOM_SEED = 42
RUN_TIME = 1440  # minutes (24 hours)
P_DIGI = 0.19

CAP_GATE_REG = 24
CAP_GATE_DIGI = 4
CAP_CHECKIN_INDIGO = 28
CAP_CHECKIN_Akasa_Air = 5
CAP_SECURITY = 55
CAP_BOARDING = 200
NUM_GATES = 7

np.random.seed(RANDOM_SEED)
start_time = time.time()

# ------------------- TIME INTERVALS -------------------
time_blocks = np.array([
    0, 60, 120, 180, 240, 300, 315, 330, 345, 360, 375, 390, 405,
    420, 435, 450, 465, 480, 495, 510, 525, 540, 600, 660, 720, 780,
    840, 900, 960, 975, 990, 1005, 1020, 1035, 1050, 1065, 1080,
    1095, 1110, 1125, 1140, 1155, 1170, 1185, 1200, 1260, 1320, 1380
])

# ------------------- REORDERED SERVICE ARRAYS -------------------
service_gate = np.array([
1.873562059,1.873562059,2.607665649,2.607665649,2.694703497,3.207665649,3.273359262,
2.875598168,3.004047443,3.757170457,4.039022931,1.607947608,1.665027568,1.607947608,
1.607947608,1.637550977,1.637550977,1.643207992,1.625980035,1.643207992,1.654275554,
1.614051694,1.595447002,1.620061696,1.548325568,1.577875003,1.595044445,1.540989546,
1.577875003,1.589409089,1.547390396,1.577875003,1.540989546,1.540989546,1.534480629,
1.56597408,1.583686703,1.589409089,1.900987697,1.909843151,1.882851225,1.918561903,
1.94393393,1.900987697,1.882851225,1.882851225,1.873562059,1.882851225,1.918561903
])

service_checkin = np.array([
2.091022075,2.777475291,2.721808294,4.037687461,4.361985378,5.659198478,5.60159215,
5.321820993,5.235947657,1.876065026,1.91119625,1.957274516,1.968305721,1.96483997,
1.948725962,1.927793286,1.936338341,1.936338341,1.948725962,1.918230719,1.881203234,
1.955236711,1.890764707,1.824491999,1.802232744,1.76626219,1.515708519,1.64136654,
1.659183649,1.64136654,1.781972235,1.675375371,1.774365785,1.789126533,1.64136654,
1.781972235,2.340199125,2.360055764,2.461985615,2.422508946,2.174891424,2.232827924,
2.038678385,2.038678385,2.174891424,2.002377287,1.802791849,2.195701591
])

service_security = np.array([
4.01075125,4.877720533,4.365247184,4.285726669,5.01075125,5.877720533,8.01075125,
9.285726669,9.12013644,3.701277027,3.701277027,3.827918394,3.701277027,3.788309462,
3.701277027,3.601809427,3.653273102,3.601809427,3.601809427,3.653273102,3.701277027,
3.653273102,3.546457181,3.567928666,3.51829196,3.567928666,3.51829196,3.567928666,
3.65747274,3.51829196,3.464874869,3.464874869,3.614203796,3.51829196,3.51829196,
3.698039578,4.539139865,4.473779257,4.40402282,4.539139865,4.329348649,4.249138637,
4.329348649,4.329348649,4.329348649,4.473779257,4.539139865,4.658390106
])

# ---------- Helper ----------
def get_service_time(service_array, t):
    idx = np.searchsorted(time_blocks, t, side="right") - 1
    idx = min(max(idx, 0), len(service_array) - 1)
    return np.random.exponential(service_array[idx])

def D_gate_Digi_T1(): return 0.5 + np.random.exponential(0.05)
def D_gate_Reg_T1(t): return get_service_time(service_gate, t)
def D_checkin_T1(t):  return get_service_time(service_checkin, t)
def D_security_T1(t): return get_service_time(service_security, t)
def D_walking():      return np.random.triangular(5.0, 10.0, 10.0)
def D_waiting():      return np.random.uniform(20.0, 40.0)
def D_boarding():     return np.random.triangular(5.0, 8.0, 10.0)

# ---------- Data ----------
df = pd.read_csv('data/Terminal2 Flight Data.csv')

# ---------- Initialize Metrics ----------
cycle_times = []
waits = {"gate_reg": [], "gate_digi": [], "checkin_indigo": [], "checkin_akasa_air": [],
         "security": [], "boarding": [], "waiting_hall": []}
queue_lengths = {"checkin": [], "security": [], "boarding": []}
busy_time = {"checkin": 0, "security": 0, "boarding": 0}
throughput = np.zeros(28)  # for 27 hours
results_by_airline = {"IndiGo": [], "Akasa Air": []}
system_count = []

# ---------- Passenger Process ----------
def passenger(env, pid, airline, gate_reg, gate_digi, checkin_indigo, checkin_akasa_air, securities, boardings):
    t0 = env.now
    system_count.append((env.now, +1))

    # Gate (Digital or Regular)
    if np.random.rand() < P_DIGI:
        chosen = min(gate_digi, key=lambda r: len(r.queue))
        with chosen.request() as req:
            t_req = env.now; yield req
            waits["gate_digi"].append(env.now - t_req)
            yield env.timeout(D_gate_Digi_T1())
    else:
        chosen = min(gate_reg, key=lambda r: len(r.queue))
        with chosen.request() as req:
            t_req = env.now; yield req
            waits["gate_reg"].append(env.now - t_req)
            yield env.timeout(D_gate_Reg_T1(env.now))

    # Check-in
    if airline.lower() == "indigo":
        chosen = min(checkin_indigo, key=lambda r: len(r.queue))
        queue_lengths["checkin"].append(len(chosen.queue))
        with chosen.request() as req:
            t_req = env.now; yield req
            waits["checkin_indigo"].append(env.now - t_req)
            st = D_checkin_T1(env.now)
            yield env.timeout(st)
            busy_time["checkin"] += st
    else:
        chosen = min(checkin_akasa_air, key=lambda r: len(r.queue))
        queue_lengths["checkin"].append(len(chosen.queue))
        with chosen.request() as req:
            t_req = env.now; yield req
            waits["checkin_akasa_air"].append(env.now - t_req)
            st = D_checkin_T1(env.now)
            yield env.timeout(st)
            busy_time["checkin"] += st

    # Security
    chosen = min(securities, key=lambda r: len(r.queue))
    queue_lengths["security"].append(len(chosen.queue))
    with chosen.request() as req:
        t_req = env.now; yield req
        waits["security"].append(env.now - t_req)
        st = D_security_T1(env.now)
        yield env.timeout(st)
        busy_time["security"] += st

    # Waiting Hall (before boarding)
    waiting_time = D_waiting()
    waits["waiting_hall"].append(waiting_time)
    yield env.timeout(D_walking() + waiting_time)

    # Boarding
    chosen = min(boardings, key=lambda r: len(r.queue))
    queue_lengths["boarding"].append(len(chosen.queue))
    with chosen.request() as req:
        t_req = env.now; yield req
        waits["boarding"].append(env.now - t_req)
        st = D_boarding()
        yield env.timeout(st)
        busy_time["boarding"] += st

    cycle_times.append(env.now - t0)
    idx = min(int(env.now // 60), len(throughput) - 1)
    throughput[idx] += 1
    results_by_airline[airline].append(env.now - t0)
    system_count.append((env.now, -1))

# ---------- Flight Source ----------
def flight_source(env, gate_reg, gate_digi, checkin_indigo, checkin_spicejet, securities, boardings):
    pid = 0
    eps = 1e-6
    for _, row in df.iterrows():
        dep_time = row["Dep_time_min"]
        n_passengers = int(row["N_passengers"])
        airline = row["Airline"]
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
        mean_arrival_gap = total_window / n_passengers
        inter_arrivals = np.random.exponential(scale=mean_arrival_gap, size=n_passengers)

        # normalize so arrivals fit inside window
        arrival_times = arr_start + np.cumsum(inter_arrivals)
        arrival_times = np.clip(arrival_times, arr_start, arr_end)
        arrival_times.sort()
        for arr_time in arrival_times:
            yield env.timeout(max(0, arr_time - env.now))
            pid += 1
            env.process(passenger(env, pid, airline, gate_reg, gate_digi, checkin_indigo, checkin_spicejet, securities, boardings))


# ---------- Simulation ----------
env = simpy.Environment()
gate_reg = [simpy.Resource(env, capacity=1) for _ in range(CAP_GATE_REG)]
gate_digi = [simpy.Resource(env, capacity=1) for _ in range(CAP_GATE_DIGI)]
checkin_indigo = [simpy.Resource(env, capacity=1) for _ in range(CAP_CHECKIN_INDIGO)]
checkin_akasa_air = [simpy.Resource(env, capacity=1) for _ in range(CAP_CHECKIN_Akasa_Air)]
securities = [simpy.Resource(env, capacity=1) for _ in range(CAP_SECURITY)]
boardings = [simpy.Resource(env, capacity=CAP_BOARDING) for _ in range(NUM_GATES)]

env.process(flight_source(env, gate_reg, gate_digi, checkin_indigo, checkin_akasa_air, securities, boardings))

print("\n--- Simulation Started ---\n")
env.run(until=RUN_TIME )

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
def safe_max(lst): return np.max(lst) if len(lst) > 0 else 0

avg_queue_len = {k: np.mean(v) if len(v) else 0 for k, v in queue_lengths.items()}
max_queue_len = {k: safe_max(v) for k, v in queue_lengths.items()}
utilization = {k: busy_time[k] / RUN_TIME for k in busy_time}
percentiles = np.percentile(cycle_times, [5, 50, 95])
system_trend = np.cumsum([c for _, c in sorted(system_count, key=lambda x: x[0])])
avg_system = np.mean(system_trend)
max_system = np.max(system_trend)

# ---------- Results ----------
print(f"\nTotal passengers completed: {len(cycle_times)}")
print(f"Average total time in system: {np.mean(cycle_times):.2f} min")
print(f"Min: {np.min(cycle_times):.2f} | Max: {np.max(cycle_times):.2f} | 95th percentile: {percentiles[2]:.2f}")

print("\n🕒 Average Waiting Times (min):")
for k, v in waits.items():
    print(f"  {k}: {avg(v):.2f}")
if "waiting_hall" in waits:
    print(f"  ➤ Waiting Hall (before boarding): avg={avg(waits['waiting_hall']):.2f}, "
          f"max={safe_max(waits['waiting_hall']):.2f}, 95th={np.percentile(waits['waiting_hall'],95):.2f}")

print("\n📈 Average Queue Lengths:", avg_queue_len)
print("📊 Max Queue Lengths:", max_queue_len)
print("⚙️ Resource Utilization:", utilization)

print("\n🚪 Throughput per Hour:")
for h in range(len(throughput)):
    print(f"{h:02d}:00 - {throughput[h]} passengers")

print("\n✈️ Breakdown by Airline:")
for a in results_by_airline:
    if len(results_by_airline[a]) > 0:
        print(f"  {a}: avg={np.mean(results_by_airline[a]):.2f}, count={len(results_by_airline[a])}")

print(f"\n👥 Average passengers in system: {avg_system:.2f}")
print(f"👥 Max passengers in system: {max_system:.2f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))
plt.plot(arrival_rate, marker='o')
plt.title("Passenger Arrival Rate per Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Passengers Arriving")
plt.grid(True)
plt.show()

"""### Passenger Throughput per Hour (Terminal 2)
Shows how many passengers were completed each hour, helping analyze overall flow efficiency.

"""

plt.figure(figsize=(10,5))
plt.plot(range(len(throughput)), throughput, marker='o', linewidth=2)
plt.title("Passenger Throughput per Hour (Terminal 2)")
plt.xlabel("Hour of Day")
plt.ylabel("Passengers Completed")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

"""### Passengers in System Over Time (Terminal 2)
Tracks the number of passengers currently being processed or waiting, over simulation time.

"""

times, deltas = zip(*sorted(system_count, key=lambda x: x[0]))
system_trend = np.cumsum(deltas)
plt.figure(figsize=(10,5))
plt.plot(times, system_trend, color='teal')
plt.title("Number of Passengers in System Over Time (T2)")
plt.xlabel("Simulation Time (min)")
plt.ylabel("Passengers in System")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

"""### Average Waiting Time per Process (Terminal 2)
Compares how long passengers wait on average at each process (check-in, security, etc.).

"""

avg_waits = {k: np.mean(v) if len(v) else 0 for k,v in waits.items()}
plt.figure(figsize=(8,5))
plt.bar(avg_waits.keys(), avg_waits.values(), color='cornflowerblue', alpha=0.8)
plt.title("Average Waiting Time per Process (T2)")
plt.ylabel("Average Waiting Time (min)")
plt.xticks(rotation=25)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

"""### Waiting Time Distribution - Security (Terminal 2)
Histogram showing variation in waiting times specifically for the security process.

"""

plt.figure(figsize=(8,5))
plt.hist(waits["security"], bins=30, color='lightcoral', alpha=0.7, edgecolor='black')
plt.title("Distribution of Waiting Times - Security (T2)")
plt.xlabel("Waiting Time (min)")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

"""### Average Queue Length per Process (Terminal 2)
Shows the mean queue length for each process — helps identify congestion points.

"""

plt.figure(figsize=(8,5))
plt.bar(queue_lengths.keys(), [np.mean(v) if v else 0 for v in queue_lengths.values()],
        color='mediumseagreen', alpha=0.8)
plt.title("Average Queue Length per Process (T2)")
plt.ylabel("Average Queue Length")
plt.xticks(rotation=25)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

"""### Resource Utilization (Terminal 2)
Displays how busy each service resource (counter/security/staff) was during the simulation.

"""

plt.figure(figsize=(8,5))
plt.bar(utilization.keys(), [v*100 for v in utilization.values()], color='goldenrod', alpha=0.9)
plt.title("Resource Utilization (%) (T2)")
plt.ylabel("Utilization (%)")
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

"""### Distribution of Total Passenger Cycle Times (Terminal 2)
Shows how long each passenger spent in the entire system (arrival → exit).

"""

plt.figure(figsize=(8,5))
plt.hist(cycle_times, bins=50, color='mediumpurple', alpha=0.75, edgecolor='black')
plt.title("Distribution of Total Passenger Cycle Times (T2)")
plt.xlabel("Total Time in System (min)")
plt.ylabel("Number of Passengers")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

"""### Average Total Time in System by Airline (Terminal 2)
Compares passenger experience across airlines to detect imbalance or inefficiency.

"""

airlines = list(results_by_airline.keys())
avg_times = [np.mean(results_by_airline[a]) if len(results_by_airline[a]) > 0 else 0 for a in airlines]
plt.figure(figsize=(8,5))
plt.bar(airlines, avg_times, color=['deepskyblue', 'orange'])
plt.title("Average Total Time in System by Airline (T2)")
plt.ylabel("Average Time (min)")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

"""### Cumulative Passengers Served Over Time (Terminal 2)
Illustrates total passengers completed as time progresses — a measure of throughput efficiency.

"""

plt.figure(figsize=(9,5))
plt.plot(np.cumsum(throughput), color='darkorange', linewidth=2)
plt.title("Cumulative Passengers Served Over Time (T2)")
plt.xlabel("Hour")
plt.ylabel("Cumulative Passengers")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

"""### Correlation between Total Waiting Time and System Time (Terminal 2)
Scatter plot showing relationship between total waiting time and total time spent in system.

"""

total_wait = []
for i in range(len(cycle_times)):
    total_wait_val = 0
    for k, v in waits.items():
        if i < len(v):
            total_wait_val += v[i]
    total_wait.append(total_wait_val)

plt.figure(figsize=(7,5))
plt.scatter(total_wait, cycle_times[:len(total_wait)], alpha=0.3, color='slateblue')
plt.title("Correlation: Total Waiting Time vs Total System Time (T2)")
plt.xlabel("Total Waiting Time (min)")
plt.ylabel("Total System Time (min)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

"""### Cycle Time Percentile Distribution (Terminal 2)
Highlights key performance percentiles (median, 75th, 90th, 95th) of total passenger time in the system.

"""

if len(cycle_times) > 0:
    percentiles = [50, 75, 90, 95]
    values = [np.percentile(cycle_times, p) for p in percentiles]
    plt.figure(figsize=(8,5))
    plt.bar([str(p)+"%" for p in percentiles], values, color='skyblue', alpha=0.8)
    plt.title("Cycle Time Percentile Distribution (T2)")
    plt.ylabel("Cycle Time (min)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

# ====================================================
# 📈 TERMINAL 2 PERFORMANCE ANALYSIS PLOTS
# ====================================================

plt.style.use('seaborn-v0_8-colorblind')

# 1️⃣ Throughput per Hour
plt.figure(figsize=(10,5))
plt.plot(range(len(throughput)), throughput, marker='o', linewidth=2)
plt.title("Passenger Throughput per Hour (Terminal 2)")
plt.xlabel("Hour of Day")
plt.ylabel("Passengers Completed")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 2️⃣ Passengers in System Over Time
times, deltas = zip(*sorted(system_count, key=lambda x: x[0]))
system_trend = np.cumsum(deltas)
plt.figure(figsize=(10,5))
plt.plot(times, system_trend, color='teal')
plt.title("Number of Passengers in System Over Time (T2)")
plt.xlabel("Simulation Time (min)")
plt.ylabel("Passengers in System")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 3️⃣ Average Waiting Time per Process
avg_waits = {k: np.mean(v) if len(v) else 0 for k,v in waits.items()}
plt.figure(figsize=(8,5))
plt.bar(avg_waits.keys(), avg_waits.values(), color='cornflowerblue', alpha=0.8)
plt.title("Average Waiting Time per Process (T2)")
plt.ylabel("Average Waiting Time (min)")
plt.xticks(rotation=25)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 4️⃣ Waiting Time Distribution - Security
plt.figure(figsize=(8,5))
plt.hist(waits["security"], bins=30, color='lightcoral', alpha=0.7, edgecolor='black')
plt.title("Distribution of Waiting Times - Security (T2)")
plt.xlabel("Waiting Time (min)")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 5️⃣ Average Queue Length per Process
plt.figure(figsize=(8,5))
plt.bar(queue_lengths.keys(), [np.mean(v) if v else 0 for v in queue_lengths.values()],
        color='mediumseagreen', alpha=0.8)
plt.title("Average Queue Length per Process (T2)")
plt.ylabel("Average Queue Length")
plt.xticks(rotation=25)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 6️⃣ Resource Utilization
plt.figure(figsize=(8,5))
plt.bar(utilization.keys(), [v*100 for v in utilization.values()], color='goldenrod', alpha=0.9)
plt.title("Resource Utilization (%) (T2)")
plt.ylabel("Utilization (%)")
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 7️⃣ Cycle Time Distribution
plt.figure(figsize=(8,5))
plt.hist(cycle_times, bins=50, color='mediumpurple', alpha=0.75, edgecolor='black')
plt.title("Distribution of Total Passenger Cycle Times (T2)")
plt.xlabel("Total Time in System (min)")
plt.ylabel("Number of Passengers")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 8️⃣ Comparison by Airline
airlines = list(results_by_airline.keys())
avg_times = [np.mean(results_by_airline[a]) if len(results_by_airline[a]) > 0 else 0 for a in airlines]
plt.figure(figsize=(8,5))
plt.bar(airlines, avg_times, color=['deepskyblue', 'orange'])
plt.title("Average Total Time in System by Airline (T2)")
plt.ylabel("Average Time (min)")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 9️⃣ Cumulative Passengers Served
plt.figure(figsize=(9,5))
plt.plot(np.cumsum(throughput), color='darkorange', linewidth=2)
plt.title("Cumulative Passengers Served Over Time (T2)")
plt.xlabel("Hour")
plt.ylabel("Cumulative Passengers")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 🔟 Waiting vs System Time Correlation
total_wait = []
for i in range(len(cycle_times)):
    # sum all waits for available lists that have enough entries
    total_wait_val = 0
    for k, v in waits.items():
        if i < len(v): total_wait_val += v[i]
    total_wait.append(total_wait_val)

plt.figure(figsize=(7,5))
plt.scatter(total_wait, cycle_times[:len(total_wait)], alpha=0.3, color='slateblue')
plt.title("Correlation: Total Waiting Time vs Total System Time (T2)")
plt.xlabel("Total Waiting Time (min)")
plt.ylabel("Total System Time (min)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

"""Terminal 1 Dummy


"""

import simpy
import numpy as np
import pandas as pd
import time
from statistics import mean

# ------------------- PARAMETERS -------------------
RANDOM_SEED = 42
RUN_TIME = 1440  # minutes (24 hours)
P_DIGI = 0.19

CAP_GATE_REG = 24
CAP_GATE_DIGI = 6
CAP_CHECKIN_INDIGO = 20
CAP_CHECKIN_SPICEJET = 10
CAP_SECURITY = 45
CAP_BOARDING = 100   # smaller so we can see waiting at boarding
NUM_GATES = 23

np.random.seed(RANDOM_SEED)
start_time = time.time()

# ------------------- TIME INTERVALS -------------------
time_blocks = np.array([
    0, 60, 120, 180, 240, 300, 315, 330, 345, 360, 375, 390, 405,
    420, 435, 450, 465, 480, 495, 510, 525, 540, 600, 660, 720, 780,
    840, 900, 960, 975, 990, 1005, 1020, 1035, 1050, 1065, 1080,
    1095, 1110, 1125, 1140, 1155, 1170, 1185, 1200, 1260, 1320, 1380
])

# ------------------- REORDERED SERVICE ARRAYS (00:00–23:59) -------------------
service_gate = np.array([
1.873562059,1.873562059,3.607665649,3.607665649,3.694703497,3.607665649,3.673359262,
3.875598168,4.004047443,3.757170457,4.039022931,1.607947608,1.665027568,1.607947608,
1.607947608,1.637550977,1.637550977,1.643207992,1.625980035,1.643207992,1.654275554,
1.614051694,1.595447002,1.620061696,1.548325568,1.577875003,1.595044445,1.540989546,
1.577875003,1.589409089,1.547390396,1.577875003,1.540989546,1.540989546,1.534480629,
1.56597408,1.583686703,1.589409089,1.900987697,1.909843151,1.882851225,1.918561903,
1.94393393,1.900987697,1.882851225,1.882851225,1.873562059,1.882851225,1.918561903
])

service_checkin = np.array([
5.091022075,3.777475291,4.721808294,5.037687461,5.361985378,5.659198478,5.60159215,
5.321820993,5.235947657,1.876065026,1.91119625,1.957274516,1.968305721,1.96483997,
1.948725962,1.927793286,1.936338341,1.936338341,1.948725962,1.918230719,1.881203234,
1.955236711,1.890764707,1.824491999,1.802232744,1.76626219,1.515708519,1.64136654,
1.659183649,1.64136654,1.781972235,1.675375371,1.774365785,1.789126533,1.64136654,
1.781972235,2.340199125,2.360055764,2.461985615,2.422508946,2.174891424,2.232827924,
2.038678385,2.038678385,2.174891424,2.002377287,1.802791849,2.195701591
])

service_security = np.array([
10.01075125,9.877720533,8.365247184,9.285726669,10.01075125,9.877720533,10.01075125,
9.285726669,9.12013644,3.701277027,3.701277027,3.827918394,3.701277027,3.788309462,
3.701277027,3.601809427,3.653273102,3.601809427,3.601809427,3.653273102,3.701277027,
3.653273102,3.546457181,3.567928666,3.51829196,3.567928666,3.51829196,3.567928666,
3.65747274,3.51829196,3.464874869,3.464874869,3.614203796,3.51829196,3.51829196,
3.698039578,4.539139865,4.473779257,4.40402282,4.539139865,4.329348649,4.249138637,
4.329348649,4.329348649,4.329348649,4.473779257,4.539139865,4.658390106
])

# ---------- Helper: sample exponential with correct mean ----------
def get_service_time(service_array, t):
    idx = np.searchsorted(time_blocks, t, side="right") - 1
    idx = min(max(idx, 0), len(service_array) - 1)
    return np.random.exponential(service_array[idx])

# ---------- Time Distributions ----------
def D_gate_Digi_T1(): return 0.5 + np.random.exponential(0.05)
def D_gate_Reg_T1(t):  return get_service_time(service_gate, t)
def D_checkin_T1(t):   return get_service_time(service_checkin, t)
def D_security_T1(t):  return get_service_time(service_security, t)
def D_walking():       return np.random.triangular(5.0, 10.0, 10.0)
def D_waiting():       return np.random.uniform(20.0, 40.0)
def D_boarding():      return np.random.triangular(5.0, 8.0, 10.0)  # increased boarding time

# ------------------- READ FLIGHT DATA -------------------
df = pd.read_csv('data/Terminal1 Flight Data.csv')

# ------------------- PASSENGER PROCESS -------------------
def passenger(env, pid, airline, gate_reg, gate_digi,
              checkin_indigo, checkin_spicejet,
              securities, boardings,
              cycle_times,
              wait_gate_reg, wait_gate_digi,
              wait_checkin_indigo, wait_checkin_spicejet,
              wait_security, wait_boarding):
    t0 = env.now

    # Gate entry
    if np.random.rand() < P_DIGI:
        chosen = min(gate_digi, key=lambda r: len(r.queue))
        with chosen.request() as req:
            t_req = env.now
            yield req
            wait_gate_digi.append(env.now - t_req)
            yield env.timeout(D_gate_Digi_T1())
    else:
        chosen = min(gate_reg, key=lambda r: len(r.queue))
        with chosen.request() as req:
            t_req = env.now
            yield req
            wait_gate_reg.append(env.now - t_req)
            yield env.timeout(D_gate_Reg_T1(env.now))

    # Airline-based check-in
    if airline.lower() == "indigo":
        chosen = min(checkin_indigo, key=lambda r: len(r.queue))
        with chosen.request() as req:
            t_req = env.now
            yield req
            wait_checkin_indigo.append(env.now - t_req)
            yield env.timeout(D_checkin_T1(env.now))
    elif airline.lower() == "spicejet":
        chosen = min(checkin_spicejet, key=lambda r: len(r.queue))
        with chosen.request() as req:
            t_req = env.now
            yield req
            wait_checkin_spicejet.append(env.now - t_req)
            yield env.timeout(D_checkin_T1(env.now))

    # Security
    chosen = min(securities, key=lambda r: len(r.queue))
    with chosen.request() as req:
        t_req = env.now
        yield req
        wait_security.append(env.now - t_req)
        yield env.timeout(D_security_T1(env.now))

    yield env.timeout(D_walking() + D_waiting())

    # Boarding
    chosen = min(boardings, key=lambda r: len(r.queue))
    with chosen.request() as req:
        t_req = env.now
        yield req
        wait_boarding.append(env.now - t_req)
        yield env.timeout(D_boarding())

    cycle_times.append(env.now - t0)

# ------------------- FLIGHT ARRIVAL GENERATOR -------------------
def flight_source(env, gate_reg, gate_digi,
                  checkin_indigo, checkin_spicejet,
                  securities, boardings,
                  cycle_times, wait_gate_reg, wait_gate_digi,
                  wait_checkin_indigo, wait_checkin_spicejet,
                  wait_security, wait_boarding, total_arrivals):

    pid = 0
    for _, row in df.iterrows():
        dep_time = row["Dep_time_min"]
        n_passengers = int(row["N_passengers"])
        airline = row["Airline"]

        if dep_time < 120:
            arrival_times = np.random.uniform(1320, 1380, n_passengers)
        else:
            arrival_times = np.random.uniform(dep_time - 120, dep_time - 60, n_passengers)

        arrival_times = np.clip(arrival_times, 0, RUN_TIME)
        arrival_times.sort()

        for arr_time in arrival_times:
            yield env.timeout(max(0, arr_time - env.now))
            pid += 1
            total_arrivals.append(1)
            env.process(passenger(env, pid, airline, gate_reg, gate_digi,
                                  checkin_indigo, checkin_spicejet,
                                  securities, boardings,
                                  cycle_times,
                                  wait_gate_reg, wait_gate_digi,
                                  wait_checkin_indigo, wait_checkin_spicejet,
                                  wait_security, wait_boarding))

# ------------------- ENVIRONMENT SETUP -------------------
env = simpy.Environment()
gate_reg = [simpy.Resource(env, capacity=1) for _ in range(CAP_GATE_REG)]
gate_digi = [simpy.Resource(env, capacity=1) for _ in range(CAP_GATE_DIGI)]
checkin_indigo = [simpy.Resource(env, capacity=1) for _ in range(CAP_CHECKIN_INDIGO)]
checkin_spicejet = [simpy.Resource(env, capacity=1) for _ in range(CAP_CHECKIN_SPICEJET)]
securities = [simpy.Resource(env, capacity=1) for _ in range(CAP_SECURITY)]
boardings = [simpy.Resource(env, capacity=CAP_BOARDING) for _ in range(NUM_GATES)]

cycle_times = []
wait_gate_reg, wait_gate_digi = [], []
wait_checkin_indigo, wait_checkin_spicejet = [], []
wait_security, wait_boarding = [], []
total_arrivals = []

env.process(flight_source(env, gate_reg, gate_digi,
                          checkin_indigo, checkin_spicejet,
                          securities, boardings,
                          cycle_times, wait_gate_reg, wait_gate_digi,
                          wait_checkin_indigo, wait_checkin_spicejet,
                          wait_security, wait_boarding, total_arrivals))

# ------------------- RUN SIMULATION -------------------
print("\n--- Simulation Started ---\n")
env.run(until=RUN_TIME + 180)
print("\n--- Simulation Completed ---")
print(f"Total runtime: {round(time.time() - start_time, 2)} seconds")

# ------------------- RESULTS -------------------
def avg(lst): return mean(lst) if lst else 0.0

print(f"\nTotal passengers arrived: {len(total_arrivals)}")
print(f"Passengers completed: {len(cycle_times)}")
print(f"Average total time in system: {mean(cycle_times):.2f} min")

print("\nAverage waiting times:")
print(f"  Gate (Regular): {avg(wait_gate_reg):.3f} min")
print(f"  Gate (DigiYatra): {avg(wait_gate_digi):.3f} min")
print(f"  Check-in (IndiGo): {avg(wait_checkin_indigo):.3f} min")
print(f"  Check-in (SpiceJet): {avg(wait_checkin_spicejet):.3f} min")
print(f"  Security: {avg(wait_security):.3f} min")
print(f"  Boarding: {avg(wait_boarding):.3f} min")

