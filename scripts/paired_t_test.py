
import simpy
import numpy as np
import pandas as pd
import time
from statistics import mean

# ------------------- PARAMETERS -------------------
def run_T1_simulation():
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
        if airline == "IndiGo":
            chosen = min(checkin_indigo, key=lambda r: len(r.queue))
            queue_lengths["checkin"].append(len(chosen.queue))
            with chosen.request() as req:
                t_req = env.now; yield req
                waits["checkin_indigo"].append(env.now - t_req)
                st = D_checkin_T1(env.now)
                yield env.timeout(st)
                busy_time["checkin"] += st
        elif airline == "SpiceJet":
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
    print("\n--- Simulation Completed ---")
    print(f"Total runtime: {round(time.time() - start_time, 2)} seconds")

    # ---------- Metrics ----------
    def avg(lst): return mean(lst) if lst else 0.0
    def safe_max(lst): return np.max(lst) if len(lst) > 0 else 0
    def safe_percentile(lst, p): return np.percentile(lst, p) if len(lst) > 0 else 0.0

    avg_queue_len = {k: np.mean(v) if len(v) else 0 for k, v in queue_lengths.items()}
    max_queue_len = {k: safe_max(v) for k, v in queue_lengths.items()}
    utilization = {k: busy_time[k] / RUN_TIME for k in busy_time}
    percentiles = safe_percentile(cycle_times, [5, 50, 95]) if cycle_times else [0,0,0]
    system_trend = np.cumsum([c for _, c in sorted(system_count, key=lambda x: x[0])]) if system_count else np.array([0])
    avg_system = np.mean(system_trend) if system_trend.size > 0 else 0.0
    max_system = np.max(system_trend) if system_trend.size > 0 else 0.0


    # ---------- Results ----------
    print(f"\nTotal passengers completed: {len(cycle_times)}")
    if cycle_times:
        print(f"Average total time in system: {np.mean(cycle_times):.2f} min")
        print(f"Min: {np.min(cycle_times):.2f} | Max: {np.max(cycle_times):.2f} | 95th percentile: {percentiles[2]:.2f}")
    else:
         print("No passengers completed simulation.")


    print("\n🕒 Average Waiting Times (min):")
    for k, v in waits.items():
        print(f"  {k}: {avg(v):.2f}")
    if "waiting_hall" in waits and waits["waiting_hall"]:
        print(f"  ➤ Waiting Hall (before boarding): avg={avg(waits['waiting_hall']):.2f}, "
              f"max={safe_max(waits['waiting_hall']):.2f}, 95th={safe_percentile(waits['waiting_hall'],95):.2f}")
    elif "waiting_hall" in waits:
        print(f"  ➤ Waiting Hall (before boarding): No data")


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
        else:
             print(f"  {a}: No passengers completed.")


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
    if cycle_times:
        report_lines.append(f"  Average : {np.mean(cycle_times):.2f}")
        report_lines.append(f"  Minimum : {np.min(cycle_times):.2f}")
        report_lines.append(f"  Maximum : {np.max(cycle_times):.2f}")
        report_lines.append(f"  95th Percentile : {safe_percentile(cycle_times, 95):.2f}")
    else:
        report_lines.append("  No passengers completed simulation.")
    report_lines.append("-"*60)

    # Section 2: Waiting times
    report_lines.append("AVERAGE WAITING TIMES (minutes)")
    for k, v in waits.items():
        report_lines.append(f"  {k:<30} {avg(v):>8.2f}")

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
    report_lines.append("AIRLINE BREAKDOWN")
    for a in results_by_airline:
        if len(results_by_airline[a]) > 0:
            print(f"  {a:<15} count={len(results_by_airline[a]):>6}, avg={np.mean(results_by_airline[a]):.2f} min, 95th={safe_percentile(results_by_airline[a], 95):.2f} min")
        else:
            print(f"  {a:<15} No passengers completed.")
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

    results = {
    # Gate entry waits
    "gate_reg": np.mean(waits["gate_reg"]) if waits["gate_reg"] else 0,
    "gate_digi": np.mean(waits["gate_digi"]) if waits["gate_digi"] else 0,

    # Check-in waits
    "checkin_indigo": np.mean(waits["checkin_indigo"]) if waits["checkin_indigo"] else 0,
    "checkin_spicejet": np.mean(waits["checkin_spicejet"]) if waits["checkin_spicejet"] else 0,

    # Security waits
    "security": np.mean(waits["security"]) if waits["security"] else 0,

    # Boarding + Waiting Hall (optional, for reference)
    "waiting_hall": np.mean(waits["waiting_hall"]) if waits["waiting_hall"] else 0,
    "boarding": np.mean(waits["boarding"]) if waits["boarding"] else 0
}


    return results

import simpy
import numpy as np
import pandas as pd
import time
from statistics import mean

def run_T2_simulation():
    # ------------------- PARAMETERS -------------------
    RANDOM_SEED = 42
    RUN_TIME = 1440  # minutes (24 hours)
    P_DIGI = 0.19

    CAP_GATE_REG = 24
    CAP_GATE_DIGI = 4
    CAP_CHECKIN_INDIGO = 28
    CAP_CHECKIN_Akasa_Air = 5
    CAP_SECURITY = 55
    CAP_BOARDING = 100
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
        elif airline.lower() == "akasa air":
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
    def flight_source(env, gate_reg, gate_digi, checkin_indigo, checkin_akasa_air, securities, boardings):
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
                env.process(passenger(env, pid, airline, gate_reg, gate_digi, checkin_indigo, checkin_akasa_air, securities, boardings))


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
    print("\n--- Simulation Completed ---")
    print(f"Total runtime: {round(time.time() - start_time, 2)} seconds")

    # ---------- Metrics ----------
    def avg(lst): return mean(lst) if lst else 0.0
    def safe_max(lst): return np.max(lst) if len(lst) > 0 else 0

    avg_queue_len = {k: np.mean(v) if len(v) else 0 for k, v in queue_lengths.items()}
    max_queue_len = {k: safe_max(v) for k, v in queue_lengths.items()}
    utilization = {k: busy_time[k] / RUN_TIME for k in busy_time}
    percentiles = np.percentile(cycle_times, [5, 50, 95]) if cycle_times else [0,0,0]
    system_trend = np.cumsum([c for _, c in sorted(system_count, key=lambda x: x[0])]) if system_count else np.array([0])
    avg_system = np.mean(system_trend) if system_trend.size > 0 else 0.0
    max_system = np.max(system_trend) if system_trend.size > 0 else 0.0

    # ---------- Results ----------
    print(f"\nTotal passengers completed: {len(cycle_times)}")
    if cycle_times:
        print(f"Average total time in system: {np.mean(cycle_times):.2f} min")
        print(f"Min: {np.min(cycle_times):.2f} | Max: {np.max(cycle_times):.2f} | 95th percentile: {percentiles[2]:.2f}")
    else:
        print("No passengers completed simulation.")

    print("\n🕒 Average Waiting Times (min):")
    for k, v in waits.items():
        print(f"  {k}: {avg(v):.2f}")
    if "waiting_hall" in waits and waits["waiting_hall"]:
        print(f"  ➤ Waiting Hall (before boarding): avg={avg(waits['waiting_hall']):.2f}, "
              f"max={safe_max(waits['waiting_hall']):.2f}, 95th={np.percentile(waits['waiting_hall'],95):.2f}")
    elif "waiting_hall" in waits:
        print(f"  ➤ Waiting Hall (before boarding): No data")


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
        else:
            print(f"  {a}: No passengers completed.")

    print(f"\n👥 Average passengers in system: {avg_system:.2f}")
    print(f"👥 Max passengers in system: {max_system:.2f}")

    # ---------- SIMAN-style formatted report (Terminal 2) ----------
    report_lines = []
    report_lines.append("\n" + "="*60)
    report_lines.append("            SIMULATION REPORT – TERMINAL 2")
    report_lines.append("="*60)
    report_lines.append(f"Total Runtime (sec): {round(time.time() - start_time, 2)}")
    report_lines.append(f"Total Passengers Completed : {len(cycle_times)}")
    report_lines.append("-"*60)

    # Section 1: Time statistics
    report_lines.append("TIME IN SYSTEM (min)")
    if cycle_times:
        report_lines.append(f"  Average : {np.mean(cycle_times):.2f}")
        report_lines.append(f"  Minimum : {np.min(cycle_times):.2f}")
        report_lines.append(f"  Maximum : {np.max(cycle_times):.2f}")
        report_lines.append(f"  95th Percentile : {np.percentile(cycle_times, 95):.2f}")
    else:
        report_lines.append("  No passengers completed simulation.")
    report_lines.append("-"*60)

    # Section 2: Waiting times
    report_lines.append("AVERAGE WAITING TIMES (minutes)")
    for k, v in waits.items():
        report_lines.append(f"  {k:<30} {avg(v):>8.2f}")

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
    report_lines.append("AIRLINE BREAKDOWN")
    for a in results_by_airline:
        if len(results_by_airline[a]) > 0:
            print(f"  {a:<15} count={len(results_by_airline[a]):>6}, avg={np.mean(results_by_airline[a]):.2f} min, 95th={np.percentile(results_by_airline[a], 95):.2f} min")
        else:
            print(f"  {a:<15} No passengers completed.")
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
    with open("Siman_Report_Terminal2.txt", "w") as f:
        f.write("\n".join(report_lines))



    results = {
    # Entry / Gate
    "gate_reg": np.mean(waits["gate_reg"]) if waits["gate_reg"] else 0,
    "gate_digi": np.mean(waits["gate_digi"]) if waits["gate_digi"] else 0,

    # Check-in waits
    "checkin_indigo": np.mean(waits["checkin_indigo"]) if waits["checkin_indigo"] else 0,
    "checkin_akasa_air": np.mean(waits["checkin_akasa_air"]) if waits["checkin_akasa_air"] else 0,

    # Security waits
    "security": np.mean(waits["security"]) if waits["security"] else 0,

    # Optional: for additional output
    "waiting_hall": np.mean(waits["waiting_hall"]) if waits["waiting_hall"] else 0,
    "boarding": np.mean(waits["boarding"]) if waits["boarding"] else 0
}



    return results

import simpy
import numpy as np
import pandas as pd
import time
from statistics import mean

def run_T3_simulation():
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
    def avg(lst): return mean(lst) if lst else 0.0 # Define avg here
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
    df = pd.read_csv('data/Terminal3 Flight Data.csv')

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
            # Passengers for International flights are either First Class (20%) or Economy (80%)
            if np.random.rand() < P_INT_FIRST:
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
        nonlocal total_entered
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
    print("\n--- Simulation Completed ---")
    print(f"Total runtime: {round(time.time() - start_time, 2)} seconds")

    # ---------- Metrics ----------
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



    results = {
    # Entry / Gate
    "gate_reg": waiting_gate_reg,
    "gate_digi": waiting_gate_digi,

    # Check-in
    "checkin": waiting_checkin,

    # Security
    "security_domestic": waiting_sec_dom,
    "security_int_first": waiting_sec_int_first,
    "security_int_econ": waiting_sec_int_econ,

    # Waiting Hall & Boarding (for reporting only)
    "waiting_hall": waiting_hall_avg,
    "boarding": boarding_queue_avg,

    # Totals (optional)
    "avg_total_time": np.mean(cycle_times) if cycle_times else 0,

    # Utilization
    "utilization_checkin": utilization["checkin"],
    "utilization_security_domestic": utilization["security_domestic"],
    "utilization_security_international": (utilization["security_int_first"] + utilization["security_int_econ"]) / 2,
}




    return results

import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np # Added import for numpy

# ================================================
# STEP 1 — Load observed wait times
# ================================================
file_path = "data/airport_data.xlsx"  # your Excel file
process_df = pd.read_excel(file_path, sheet_name="process_times")

# Strip spaces from all column names
process_df.columns = [c.strip() for c in process_df.columns]

# Keep only relevant columns and rename
if 'Avg_Wait_Time_min' in process_df.columns:
    process_df.rename(columns={'Avg_Wait_Time_min':'Observed_Wait'}, inplace=True)

# Check if 'Subtype' column exists before trying to use it for merging
if 'Subtype' in process_df.columns:
    process_df = process_df[['Terminal', 'Process', 'Subtype', 'Observed_Wait']].copy()
else:
    # If no Subtype column, create a dummy one filled with NaN for merging consistency
    process_df = process_df[['Terminal', 'Process', 'Observed_Wait']].copy()
    process_df['Subtype'] = np.nan


print(process_df.head())

# ================================================
# STEP 2 — Helper functions to calculate simulated wait
# ================================================
def avg_checkin(sim, terminal):
    """Return average check-in wait for a terminal"""
    if terminal == 'T1':
        keys = ['checkin_indigo', 'checkin_spicejet']
    elif terminal == 'T2':
        keys = ['checkin_indigo', 'checkin_akasa_air']
    else:  # T3
        keys = ['checkin']
    # Correctly retrieve and average the wait times
    valid_waits = [sim.get(k, 0) for k in keys if sim.get(k, 0) > 0]
    return np.mean(valid_waits) if valid_waits else 0

def avg_security(sim, terminal, subtype):
    """Return security wait based on terminal and subtype"""

    if terminal in ['T1', 'T2']:
        return sim.get('security', 0)  # T1 and T2 have one security type

    # ---- Terminal 3 below ----
    if terminal == 'T3':
        if subtype == 'Domestic':
            return sim.get('security_domestic', 0)

        elif subtype == 'International':
            # Average of international first + economy if available
            waits = [
                sim.get('security_int_first', 0),
                sim.get('security_int_econ', 0)
            ]
            waits = [w for w in waits if w > 0]
            return np.mean(waits) if waits else 0

    return 0


def avg_entry(sim, subtype):
    """Return average entry wait based on subtype"""
    if subtype == 'Regular':
        return sim.get('gate_reg', 0)
    elif subtype == 'DigiYatra':
        return sim.get('gate_digi', 0)
    return 0


# ================================================
# STEP 3 — Run your simulation functions
# ================================================
# Example: call your existing functions
sim_T1 = run_T1_simulation()  # returns dict with metrics
sim_T2 = run_T2_simulation()  # similar
sim_T3 = run_T3_simulation()  # similar

# Combine simulation outputs into a DataFrame
sim_results = pd.DataFrame([
    # T1
    {'Terminal':'T1','Process':'Check-In','Subtype':np.nan,'Avg_Wait':avg_checkin(sim_T1,'T1')},
    {'Terminal':'T1','Process':'Entry','Subtype':'Regular','Avg_Wait':avg_entry(sim_T1,'Regular')},
    {'Terminal':'T1','Process':'Entry','Subtype':'DigiYatra','Avg_Wait':avg_entry(sim_T1,'DigiYatra')},
    {'Terminal':'T1','Process':'Security','Subtype':'Domestic','Avg_Wait':avg_security(sim_T1,'T1','Domestic')},

    # T2
    {'Terminal':'T2','Process':'Check-In','Subtype':np.nan,'Avg_Wait':avg_checkin(sim_T2,'T2')},
    {'Terminal':'T2','Process':'Entry','Subtype':'Regular','Avg_Wait':avg_entry(sim_T2,'Regular')},
    {'Terminal':'T2','Process':'Entry','Subtype':'DigiYatra','Avg_Wait':avg_entry(sim_T2,'DigiYatra')},
    {'Terminal':'T2','Process':'Security','Subtype':'Domestic','Avg_Wait':avg_security(sim_T2,'T2','Domestic')},

    # T3
    {'Terminal':'T3','Process':'Check-In','Subtype':np.nan,'Avg_Wait':avg_checkin(sim_T3,'T3')},
    {'Terminal':'T3','Process':'Entry','Subtype':'Regular','Avg_Wait':avg_entry(sim_T3,'Regular')},
    {'Terminal':'T3','Process':'Entry','Subtype':'DigiYatra','Avg_Wait':avg_entry(sim_T3,'DigiYatra')},
    {'Terminal':'T3','Process':'Security','Subtype':'Domestic','Avg_Wait':avg_security(sim_T3,'T3','Domestic')},
    {'Terminal':'T3','Process':'Security','Subtype':'International','Avg_Wait':avg_security(sim_T3,'T3','International')}
])


# ================================================
# STEP 4 — Merge observed and simulated
# ================================================
merged = pd.merge(sim_results, process_df, on=['Terminal','Process','Subtype'], how='left')
print("\n✅ Observed vs Simulated Wait Times:")
print(merged)

# ================================================
# STEP 5 — Paired t-test
# ================================================
merged_for_ttest = merged.dropna(subset=['Observed_Wait', 'Avg_Wait'])

if len(merged_for_ttest) > 1:
    t_stat, p_value = stats.ttest_rel(merged_for_ttest['Observed_Wait'], merged_for_ttest['Avg_Wait'])
    print("\nPaired t-test results:")
    print(f"T-statistic = {t_stat:.3f}, P-value = {p_value:.3f}")
    if p_value > 0.05:
        print("✅ Model validated (no significant difference).")
    else:
        print("❌ Model not validated (significant difference).")
else:
    print("\nInsufficient data points for paired t-test after dropping NaNs.")


# ================================================
# STEP 6 — Bar chart for PPT
# ================================================
plt.figure(figsize=(12,6)) # Increased figure size for better readability
bar_width = 0.35
labels = merged.apply(lambda row: f"{row['Terminal']} - {row['Process']} ({row['Subtype']})" if pd.notna(row['Subtype']) else f"{row['Terminal']} - {row['Process']}", axis=1)


x = np.arange(len(labels)) # the label locations

plt.bar(x - bar_width/2, merged['Observed_Wait'], width=bar_width, label='Observed', alpha=0.7)
plt.bar(x + bar_width/2, merged['Avg_Wait'], width=bar_width, label='Simulated', alpha=0.7)

plt.ylabel("Average Wait Time (min)")
plt.title("Observed vs Simulated Wait Times")
plt.xticks(x, labels, rotation=45, ha='right') # Set tick locations and labels
plt.legend()
plt.tight_layout()
plt.show()