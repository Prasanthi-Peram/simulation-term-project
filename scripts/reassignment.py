
"""Terminal 1

"""

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
    df = pd.read_csv('data/Terminal1 Flight Data.csv', skipinitialspace=True)
    # Clean up: remove empty columns and ensure proper data types
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = df.columns.str.strip()
    df['Dep_time_min'] = pd.to_numeric(df['Dep_time_min'], errors='coerce').fillna(0)
    df['N_passengers'] = pd.to_numeric(df['N_passengers'], errors='coerce').fillna(0).astype(int)
    df['Airline'] = df['Airline'].astype(str).str.strip()

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
            report_lines.append(f"  {a:<15} count={len(results_by_airline[a]):>6}, avg={np.mean(results_by_airline[a]):.2f} min, 95th={np.percentile(results_by_airline[a], 95):.2f} min")
        else:
            report_lines.append(f"  {a:<15} No passengers completed.")
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
        "avg_total_time": np.mean(cycle_times) if cycle_times else 0,
        "avg_wait_checkin": avg(waits["checkin_indigo"] + waits["checkin_spicejet"]),
        "avg_wait_security_dom": avg(waits["security"]), # Assuming all security is domestic for T1
        "avg_wait_security_int": 0, # No international security in T1
        "utilization_checkin": utilization["checkin"],
        "utilization_security_dom": utilization["security"],
        "utilization_security_int": 0, # No international security in T1
    }

    return results

"""Terminal 2"""

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
    df = pd.read_csv('data/Terminal2 Flight Data.csv', skipinitialspace=True)
    # Clean up: remove empty columns and ensure proper data types
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = df.columns.str.strip()
    df['Dep_time_min'] = pd.to_numeric(df['Dep_time_min'], errors='coerce').fillna(0)
    df['N_passengers'] = pd.to_numeric(df['N_passengers'], errors='coerce').fillna(0).astype(int)
    df['Airline'] = df['Airline'].astype(str).str.strip()

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
        "avg_total_time": np.mean(cycle_times) if cycle_times else 0,
        "avg_wait_checkin": avg(waits["checkin_indigo"] + waits["checkin_akasa_air"]),
        "avg_wait_security_dom": avg(waits["security"]), # Assuming all security is domestic for T2
        "avg_wait_security_int": 0, # No international security in T2
        "utilization_checkin": utilization["checkin"],
        "utilization_security_dom": utilization["security"],
        "utilization_security_int": 0, # No international security in T2
    }

    return results

"""Terminal 3"""

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
        "avg_total_time": np.mean(cycle_times) if cycle_times else 0,
        "avg_wait_checkin": waiting_checkin,
        "avg_wait_security_dom": waiting_sec_dom,
        "avg_wait_security_int": avg(waits["security_int_first"] + waits["security_int_econ"]),
        "utilization_checkin": utilization["checkin"],
        "utilization_security_dom": utilization["security_domestic"],
        "utilization_security_int": (utilization["security_int_first"] + utilization["security_int_econ"]) / 2 if utilization["security_int_first"] or utilization["security_int_econ"] else 0, # Corrected average calculation
    }

    return results

# pip install pulp

"""**Objective**

The goal of this model is to redistribute flights originally assigned to Terminal 2 (T2) between Terminal 1 (T1) and Terminal 3 (T3) to achieve a balanced passenger load and minimal overall average waiting time, considering capacity and congestion effects.
"""

import os
import pandas as pd
import numpy as np

# ------------------ USER CONFIG ------------------
T1_FILE_PATH = 'data/Terminal1 Flight Data.csv'
T2_FILE_PATH = 'data/Terminal2 Flight Data.csv'
T3_FILE_PATH = 'data/Terminal3 Flight Data.csv'
G_T1 = 23
G_T3 = 38
STAY_TIME = 30   # minutes a flight occupies a gate
ARRIVAL_LEAD_MIN = 30  # flights arrive ARRIVAL_LEAD_MIN before departure
DAY_END_MIN = 24 * 60

# Baseline waits (before reassignment averages per passenger for each terminal)
BASE_WAITS = {"T1": 96.28792715138799, "T2": 119.74782922470882, "T3": 117.46491843952965}

# ------------------ HELPERS ------------------
def find_time_col(df):
    for c in df.columns:
        low = str(c).lower()
        if 'dep' in low and ('time' in low or 'min' in low):
            return c
    for c in df.columns:
        low = str(c).lower()
        if 'departure' in low or 'dep_time' in low or 'etd' in low:
            return c
    for c in df.columns:
        low = str(c).lower()
        if 'time' in low:
            return c
    return None

def find_pass_col(df):
    for c in df.columns:
        low = str(c).lower()
        if 'pass' in low or 'pax' in low:
            return c
    return None

def find_flight_col(df):
    for c in df.columns:
        low = str(c).lower()
        if 'flight' in low:
            return c
    return None

def adjusted_wait_gate(base_wait, occupied_gates, total_gates):
    """wait = base_wait / (1 - utilization)^2 ; utilization = occupied/total (capped at 0.99)"""
    if total_gates <= 0:
        return base_wait * 5.0
    util = occupied_gates / total_gates
    util = min(util, 0.99)
    denom = (1.0 - util) ** 2
    return base_wait / max(1e-9, denom)

# ------------------ READ INPUT ------------------
if not os.path.exists(T1_FILE_PATH):
    raise FileNotFoundError(f"File not found at {T1_FILE_PATH}. Upload file or adjust FILE_PATH.")
if not os.path.exists(T2_FILE_PATH):
    raise FileNotFoundError(f"File not found at {T2_FILE_PATH}. Upload file or adjust FILE_PATH.")
if not os.path.exists(T3_FILE_PATH):
    raise FileNotFoundError(f"File not found at {T3_FILE_PATH}. Upload file or adjust FILE_PATH.")

print("✅ Reading CSV files...")

t1_df = pd.read_csv(T1_FILE_PATH, skipinitialspace=True)
t2_df = pd.read_csv(T2_FILE_PATH, skipinitialspace=True)
t3_df = pd.read_csv(T3_FILE_PATH, skipinitialspace=True)

# Normalize column names and clean data
def clean_csv_dataframe(df):
    """Clean and normalize CSV dataframe"""
    # Remove empty/Unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = [str(c).strip() for c in df.columns]
    # Ensure proper data types
    if 'Dep_time_min' in df.columns:
        df['Dep_time_min'] = pd.to_numeric(df['Dep_time_min'], errors='coerce').fillna(0)
    if 'N_passengers' in df.columns:
        df['N_passengers'] = pd.to_numeric(df['N_passengers'], errors='coerce').fillna(0).astype(int)
    if 'Airline' in df.columns:
        df['Airline'] = df['Airline'].astype(str).str.strip()
    if 'Type' in df.columns:
        df['Type'] = df['Type'].astype(str).str.strip()
    return df

t1_df = clean_csv_dataframe(t1_df)
t2_df = clean_csv_dataframe(t2_df)
t3_df = clean_csv_dataframe(t3_df)

# ------------------ NORMALIZE & CREATE ARRIVAL TIMES ------------------
def prepare_df_for_schedule(df, terminal_label):
    time_col = find_time_col(df)
    pass_col = find_pass_col(df)
    flight_col = find_flight_col(df)

    out = df.copy()

    # Flight_number
    if flight_col is None:
        out['Flight_number'] = terminal_label + "_" + out.index.astype(str)
    else:
        out['Flight_number'] = out[flight_col].astype(str)

    # Passenger count
    if pass_col is None:
        out['N_passengers'] = 180
    else:
        out['N_passengers'] = pd.to_numeric(out[pass_col], errors='coerce').fillna(180).astype(int)

    # Departure time to minutes
    if time_col is None:
        out['Dep_time_min'] = np.random.randint(0, DAY_END_MIN, size=len(out))
    else:
        def to_min_cell(v):
            if pd.isnull(v):
                return 0
            if isinstance(v, str) and ':' in v:
                try:
                    h, m = v.split(':')
                    return int(h) * 60 + int(m)
                except:
                    try:
                        return int(float(v))
                    except:
                        return 0
            try:
                return int(float(v))
            except:
                return 0
        out['Dep_time_min'] = out[time_col].apply(to_min_cell).astype(int)

    # Arrival lead
    out['Arr_time_min'] = (out['Dep_time_min'] - ARRIVAL_LEAD_MIN).clip(lower=0).astype(int)
    out['Original_Terminal'] = terminal_label
    return out[['Flight_number', 'Dep_time_min', 'Arr_time_min', 'N_passengers', 'Original_Terminal']]

t1_sched = prepare_df_for_schedule(t1_df, 'T1')
t2_sched = prepare_df_for_schedule(t2_df, 'T2')
t3_sched = prepare_df_for_schedule(t3_df, 'T3')

# ------------------ BUILD CHRONOLOGICAL EVENT LIST ------------------
all_flights = pd.concat([t1_sched, t3_sched, t2_sched], ignore_index=True)
all_flights['Flexible'] = all_flights['Original_Terminal'] == 'T2'
all_flights = all_flights.sort_values(['Arr_time_min', 'Dep_time_min']).reset_index(drop=True)

def run_scenario(assign_mode):
    """
    assign_mode: 'optimized', 'all_to_t1', 'all_to_t3', '50_50'
    Returns: resulting all_flights dataframe + T1/T3 avg wait + passenger loads
    """

    # Recreate gates fresh
    T1_gates = [0] * G_T1
    T3_gates = [0] * G_T3

    # Copy flight table fresh
    df = all_flights.copy()
    df['Assigned_Terminal'] = None
    df['Predicted_Wait_Assigned'] = np.nan

    t2_counter = 0  # for alternating assignment

    for idx, row in df.iterrows():
        arr = int(row['Arr_time_min'])
        pax = int(row['N_passengers'])
        origin = row['Original_Terminal']

        # Occupancy snapshot
        occ_T1 = sum(1 for g in T1_gates if g > arr)
        occ_T3 = sum(1 for g in T3_gates if g > arr)
        wait_T1 = adjusted_wait_gate(BASE_WAITS['T1'], occ_T1, G_T1)
        wait_T3 = adjusted_wait_gate(BASE_WAITS['T3'], occ_T3, G_T3)

        # Gate availability
        t1_free = min(T1_gates) <= arr
        t3_free = min(T3_gates) <= arr

        # -------- Assignment Rule --------
        if origin in ['T1', 'T3'] or assign_mode == 'optimized':
            # Use existing logic
            if origin == 'T1':
                assigned = 'T1'
            elif origin == 'T3':
                assigned = 'T3'
            else:
                # This is your original optimized decision:
                if t1_free and t3_free:
                    assigned = 'T1' if wait_T1 <= wait_T3 else 'T3'
                elif t1_free:
                    assigned = 'T1'
                elif t3_free:
                    assigned = 'T3'
                else:
                    assigned = 'T1' if min(T1_gates) <= min(T3_gates) else 'T3'

        elif assign_mode == 'all_to_t1':
            assigned = 'T1'

        elif assign_mode == 'all_to_t3':
            assigned = 'T3'

        elif assign_mode == '50_50':
            assigned = 'T1' if (t2_counter % 2 == 0) else 'T3'
            t2_counter += 1
        # ---------------------------------

        # Schedule gate
        if assigned == 'T1':
            start = arr if t1_free else min(T1_gates)
            gate = T1_gates.index(min(T1_gates))
            T1_gates[gate] = start + STAY_TIME
            used_wait = wait_T1
        else:
            start = arr if t3_free else min(T3_gates)
            gate = T3_gates.index(min(T3_gates))
            T3_gates[gate] = start + STAY_TIME
            used_wait = wait_T3

        df.at[idx, 'Assigned_Terminal'] = assigned
        df.at[idx, 'Predicted_Wait_Assigned'] = used_wait

    # Compute weighted averages
    assigned_T1 = df[df['Assigned_Terminal']=='T1']
    assigned_T3 = df[df['Assigned_Terminal']=='T3']

    avg_T1 = (assigned_T1['N_passengers']*assigned_T1['Predicted_Wait_Assigned']).sum() / assigned_T1['N_passengers'].sum()
    avg_T3 = (assigned_T3['N_passengers']*assigned_T3['Predicted_Wait_Assigned']).sum() / assigned_T3['N_passengers'].sum()

    return df, avg_T1, avg_T3


# Add columns for results
all_flights['Assigned_Terminal'] = None
all_flights['Scheduled_Start'] = None
all_flights['Gate_Index'] = None
all_flights['Predicted_Wait_Assigned'] = np.nan  # track per-flight wait actually used at decision time

# ------------------ INITIALIZE GATES ------------------
T1_gates = [0] * G_T1  # next-free time
T3_gates = [0] * G_T3
running_load_T1 = 0
running_load_T3 = 0

# ------------------ MAIN SCHEDULER LOOP ------------------
for idx, row in all_flights.iterrows():
    arr = int(row['Arr_time_min'])
    pax = int(row['N_passengers'])
    origin = row['Original_Terminal']

    # occupancy snapshot at 'arr'
    occupied_T1 = sum(1 for g in T1_gates if g > arr)
    occupied_T3 = sum(1 for g in T3_gates if g > arr)

    # predicted waits now
    wait_T1_now = adjusted_wait_gate(BASE_WAITS['T1'], occupied_T1, G_T1)
    wait_T3_now = adjusted_wait_gate(BASE_WAITS['T3'], occupied_T3, G_T3)

    # gate availability now
    t1_free_now = min(T1_gates) <= arr
    t3_free_now = min(T3_gates) <= arr

    if origin == 'T1':
        start_time = arr if t1_free_now else min(T1_gates)
        gate_idx = T1_gates.index(min(T1_gates))
        T1_gates[gate_idx] = start_time + STAY_TIME
        assigned = 'T1'
        running_load_T1 += pax
        used_wait = wait_T1_now

    elif origin == 'T3':
        start_time = arr if t3_free_now else min(T3_gates)
        gate_idx = T3_gates.index(min(T3_gates))
        T3_gates[gate_idx] = start_time + STAY_TIME
        assigned = 'T3'
        running_load_T3 += pax
        used_wait = wait_T3_now

    else:
        # T2: decide between T1 and T3
        if t1_free_now and t3_free_now:
            assigned = 'T1' if wait_T1_now <= wait_T3_now else 'T3'
            start_time = arr
        elif t1_free_now:
            assigned = 'T1'; start_time = arr
        elif t3_free_now:
            assigned = 'T3'; start_time = arr
        else:
            next_free_T1 = min(T1_gates)
            next_free_T3 = min(T3_gates)
            if next_free_T1 < next_free_T3:
                assigned = 'T1'; start_time = next_free_T1
            elif next_free_T3 < next_free_T1:
                assigned = 'T3'; start_time = next_free_T3
            else:
                # both free at same time; compare predicted waits at that time
                occ_T1_then = sum(1 for g in T1_gates if g > next_free_T1)
                occ_T3_then = sum(1 for g in T3_gates if g > next_free_T3)
                wait_T1_then = adjusted_wait_gate(BASE_WAITS['T1'], occ_T1_then, G_T1)
                wait_T3_then = adjusted_wait_gate(BASE_WAITS['T3'], occ_T3_then, G_T3)
                assigned = 'T1' if wait_T1_then <= wait_T3_then else 'T3'
                start_time = next_free_T1

        if assigned == 'T1':
            gate_idx = T1_gates.index(min(T1_gates))
            T1_gates[gate_idx] = start_time + STAY_TIME
            running_load_T1 += pax
            used_wait = wait_T1_now
        else:
            gate_idx = T3_gates.index(min(T3_gates))
            T3_gates[gate_idx] = start_time + STAY_TIME
            running_load_T3 += pax
            used_wait = wait_T3_now

    # record
    all_flights.at[idx, 'Assigned_Terminal'] = assigned
    all_flights.at[idx, 'Scheduled_Start'] = int(start_time)
    all_flights.at[idx, 'Gate_Index'] = int(gate_idx)
    all_flights.at[idx, 'Predicted_Wait_Assigned'] = used_wait

# ------------------ FINAL STATS & SAVE ------------------
# End-of-day gates busy (informational)
occupied_after_day_T1 = sum(1 for g in T1_gates if g > DAY_END_MIN)
occupied_after_day_T3 = sum(1 for g in T3_gates if g > DAY_END_MIN)

# ------------------ BEFORE vs AFTER (T1 & T3) ------------------
# BEFORE: baseline averages per passenger for T1 and T3
before_T1 = BASE_WAITS['T1']
before_T3 = BASE_WAITS['T3']

# AFTER: passenger-weighted averages for flights actually assigned to T1 and T3
assigned_T1 = all_flights[all_flights['Assigned_Terminal'] == 'T1'].dropna(subset=['Predicted_Wait_Assigned'])
assigned_T3 = all_flights[all_flights['Assigned_Terminal'] == 'T3'].dropna(subset=['Predicted_Wait_Assigned'])

def weighted_avg_wait(df):
    tot = df['N_passengers'].sum()
    if tot <= 0:
        return float('nan')
    return (df['N_passengers'] * df['Predicted_Wait_Assigned']).sum() / tot

after_T1 = weighted_avg_wait(assigned_T1)
after_T3 = weighted_avg_wait(assigned_T3)

print("\n=== AVERAGE WAITING TIME PER PASSENGER (T1 & T3) ===")
print(f"T1 - Before reassignment: {before_T1:.2f} minutes")
print(f"T1 - After  reassignment: {after_T1:.2f} minutes")
print(f"T1 - Change: {after_T1 - before_T1:+.2f} minutes per passenger")

print(f"\nT3 - Before reassignment: {before_T3:.2f} minutes")
print(f"T3 - After  reassignment: {after_T3:.2f} minutes")
print(f"T3 - Change: {after_T3 - before_T3:+.2f} minutes per passenger")

# Additional context (optional prints)
pax_T1_after = assigned_T1['N_passengers'].sum()
pax_T3_after = assigned_T3['N_passengers'].sum()
print(f"\nPassengers handled AFTER reassignment -> T1: {pax_T1_after:,} | T3: {pax_T3_after:,}")
print(f"Gates busy past day end -> T1: {occupied_after_day_T1}, T3: {occupied_after_day_T3}")

# ------------------ RATIOS OF ASSIGNMENT (T2 → T1/T3) ------------------
t2_reassigned = all_flights[all_flights['Original_Terminal'] == 'T2']

# Flight counts
count_T1 = (t2_reassigned['Assigned_Terminal'] == 'T1').sum()
count_T3 = (t2_reassigned['Assigned_Terminal'] == 'T3').sum()

# Passenger counts
pax_T1 = t2_reassigned.loc[t2_reassigned['Assigned_Terminal'] == 'T1', 'N_passengers'].sum()
pax_T3 = t2_reassigned.loc[t2_reassigned['Assigned_Terminal'] == 'T3', 'N_passengers'].sum()

total_flights = count_T1 + count_T3
total_pax = pax_T1 + pax_T3

print("\n=== REASSIGNMENT RATIOS (T2 → T1 / T3) ===")
print(f"Flights to T1: {count_T1}  |  Flights to T3: {count_T3}")
if total_flights > 0:
    print(f"Flight Ratio (T1 : T3) = {count_T1/total_flights:.2f} : {count_T3/total_flights:.2f}")

print(f"\nPassengers to T1: {pax_T1}  |  Passengers to T3: {pax_T3}")
if total_pax > 0:
    print(f"Passenger Ratio (T1 : T3) = {pax_T1/total_pax:.2f} : {pax_T3/total_pax:.2f}")


# Save outputs
t2_only_assigned = all_flights[all_flights['Original_Terminal']=='T2'].copy()
out_t2 = 'data/T2_gate_assignment_full.xlsx'
out_all = 'data/All_terminals_full_schedule.xlsx'
t2_only_assigned.to_excel(out_t2, index=False)
all_flights.to_excel(out_all, index=False)
print(f"\n✅ Saved T2 assignment to: {out_t2}")
print(f"✅ Saved full schedule to: {out_all}")
print("\n=== SCHEDULING COMPLETE ===")

scenarios = ["optimized", "all_to_t1", "all_to_t3", "50_50"]

scenario_results = {}  # store results
for mode in scenarios:
    df_res, avg1, avg3 = run_scenario(mode)
    scenario_results[mode] = (df_res, avg1, avg3)
    print(f"\n=== SCENARIO: {mode} ===")
    print(f"T1 Avg Wait: {avg1:.2f} min")
    print(f"T3 Avg Wait: {avg3:.2f} min")

import matplotlib.pyplot as plt
# labels was not defined, use 'scenarios' instead
labels = scenarios

combined_avgs = []
for mode in scenarios:
    df = scenario_results[mode][0]
    # Weighted average across T1+T3 together
    total_pax = df['N_passengers'].sum()
    combined_wait = (df['N_passengers'] * df['Predicted_Wait_Assigned']).sum() / total_pax
    combined_avgs.append(combined_wait)

plt.figure()
plt.bar(labels, combined_avgs)
plt.ylabel("Overall Avg Waiting Time (min)")
plt.title("Combined (T1 + T3) Average Waiting Time Across Scenarios")
plt.show()

import numpy as np

labels = scenarios # Define labels
t1_avgs = [scenario_results[mode][1] for mode in scenarios]
t3_avgs = [scenario_results[mode][2] for mode in scenarios]

x = np.arange(len(scenarios))
width = 0.35

plt.figure(figsize=(8,5))
plt.bar(x - width/2, t1_avgs, width, label="T1")
plt.bar(x + width/2, t3_avgs, width, label="T3")

plt.xticks(x, labels)
plt.ylabel("Average Waiting Time (min)")
plt.title("Average Waiting Time Comparison Across Scenarios")
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

labels = scenarios # Define labels

t1_loads = [scenario_results[mode][0][scenario_results[mode][0]['Assigned_Terminal']=='T1']['N_passengers'].sum() for mode in scenarios]
t3_loads = [scenario_results[mode][0][scenario_results[mode][0]['Assigned_Terminal']=='T3']['N_passengers'].sum() for mode in scenarios]

x = np.arange(len(scenarios))
width = 0.35

plt.figure(figsize=(8,5))
plt.bar(x - width/2, t1_loads, width, label="T1")
plt.bar(x + width/2, t3_loads, width, label="T3")

plt.xticks(x, labels)
plt.ylabel("Total Passengers Handled")
plt.title("Passenger Distribution Across Scenarios")
plt.legend()
plt.show()

def compute_gate_usage(df):
    occ_T1 = [0]*(DAY_END_MIN+1)
    occ_T3 = [0]*(DAY_END_MIN+1)
    for i,row in df.iterrows():
        start = row['Arr_time_min']
        end = start + STAY_TIME
        if row['Assigned_Terminal']=='T1':
            for t in range(start, min(end, DAY_END_MIN)):
                occ_T1[t]+=1
        else:
            for t in range(start, min(end, DAY_END_MIN)):
                occ_T3[t]+=1
    return occ_T1, occ_T3

plt.figure()
for mode in scenarios:
    df = scenario_results[mode][0]
    occ1, _ = compute_gate_usage(df)
    plt.plot(occ1, label=mode)
plt.xlabel("Minute of Day")
plt.ylabel("Gates Occupied")
plt.title("Terminal 1 Gate Congestion Over Time")
plt.legend()
plt.show()

plt.figure()
for mode in scenarios:
    df = scenario_results[mode][0]
    _, occ3 = compute_gate_usage(df)
    plt.plot(occ3, label=mode)
plt.xlabel("Minute of Day")
plt.ylabel("Gates Occupied")
plt.title("Terminal 3 Gate Congestion Over Time")
plt.legend()
plt.show()

pax_t1_5050 = scenario_results["50_50"][0][scenario_results["50_50"][0]['Assigned_Terminal']=="T1"]['N_passengers'].sum()
pax_t3_5050 = scenario_results["50_50"][0][scenario_results["50_50"][0]['Assigned_Terminal']=="T3"]['N_passengers'].sum()

plt.figure()
plt.pie([pax_t1_5050, pax_t3_5050], autopct='%1.1f%%', labels=['T1','T3'])
plt.title("Passenger Share Split in 50-50 Scenario")
plt.show()

import matplotlib.pyplot as plt
import numpy as np

HORIZON = 24*60
occ_T1 = np.zeros(HORIZON, dtype=int)
occ_T3 = np.zeros(HORIZON, dtype=int)

for _, r in all_flights.iterrows():
    s = int(r['Scheduled_Start'])
    e = min(s + STAY_TIME, HORIZON)
    if r['Assigned_Terminal'] == 'T1':
        occ_T1[s:e] += 1
    else:
        occ_T3[s:e] += 1

plt.figure(figsize=(13,4))
plt.plot(occ_T1, label='T1 Gates Used')
plt.plot(occ_T3, label='T3 Gates Used')
plt.axhline(G_T1, linestyle='--', label='T1 Capacity')
plt.axhline(G_T3, linestyle='--', label='T3 Capacity')
plt.title('Gate Occupancy Over Time')
plt.xlabel('Minutes of Day')
plt.ylabel('Occupied Gates')
plt.legend()
plt.show()

wait_T1_curve = [adjusted_wait_gate(BASE_WAITS['T1'], g, G_T1) for g in occ_T1]
wait_T3_curve = [adjusted_wait_gate(BASE_WAITS['T3'], g, G_T3) for g in occ_T3]

plt.figure(figsize=(13,4))
plt.plot(wait_T1_curve, label='T1 Predicted Wait')
plt.plot(wait_T3_curve, label='T3 Predicted Wait')
plt.title('Predicted Wait Over Time')
plt.xlabel('Minutes of Day')
plt.ylabel('Wait (minutes)')
plt.legend()
plt.show()

# --- FLIGHT SPLIT (T2 Flights Only) ---
t2_only = all_flights[all_flights['Original_Terminal'] == 'T2']
flight_split = t2_only['Assigned_Terminal'].value_counts().reindex(['T1','T3']).fillna(0)

plt.figure(figsize=(6,5))
plt.pie(flight_split, labels=flight_split.index, autopct='%1.1f%%', startangle=90)
plt.title('Reassigned T2 Flights Split (T1 vs T3)')
plt.show()

print("\nT2 Flight Reassignment Counts:")
print(flight_split)

ratio_T1 = flight_split['T1']
ratio_T3 = flight_split['T3']
print(f"\nT2 Flight Ratio (T1 : T3) = {ratio_T1} : {ratio_T3}")

pax_counts = all_flights.groupby('Assigned_Terminal')['N_passengers'].sum()
pax_counts.plot(kind='bar', figsize=(5,4), title='Passenger Load Distribution (T1 vs T3)')
plt.ylabel('Passengers')
plt.show()

all_flights['Hour'] = all_flights['Dep_time_min'] // 60
hourly = all_flights.groupby(['Hour', 'Assigned_Terminal']).size().unstack(fill_value=0)

hourly.plot(kind='bar', stacked=True, figsize=(12,4))
plt.title('Hourly Departure Volume by Assigned Terminal')
plt.ylabel('Flights')
plt.show()

plt.figure(figsize=(7,4))
plt.hist(all_flights['Predicted_Wait_Assigned'], bins=30)
plt.title('Distribution of Assigned Predicted Wait Times')
plt.xlabel('Wait (minutes)')
plt.ylabel('Number of Flights')
plt.show()

import matplotlib.pyplot as plt

# ------------------ BEFORE vs AFTER WAITING TIME ------------------
before_waits = [BASE_WAITS['T1'], BASE_WAITS['T3']]
after_waits = [after_T1, after_T3]

plt.figure(figsize=(7,5))
x = np.arange(2)
width = 0.35

plt.bar(x - width/2, before_waits, width, label='Before')
plt.bar(x + width/2, after_waits, width, label='After')

plt.xticks(x, ['T1', 'T3'])
plt.ylabel('Average Passenger Waiting Time (minutes)')
plt.title('Before vs After Reassignment - Waiting Time')
plt.legend()
plt.tight_layout()
plt.show()


# ------------------ BEFORE vs AFTER PASSENGER LOAD ------------------
# Passengers originally at T1 and T3 (before reassignment)
pax_T1_before = t1_sched['N_passengers'].sum()
pax_T3_before = t3_sched['N_passengers'].sum()

pax_before = [pax_T1_before, pax_T3_before]
pax_after = [pax_T1_after, pax_T3_after]

plt.figure(figsize=(7,5))
plt.bar(x - width/2, pax_before, width, label='Before')
plt.bar(x + width/2, pax_after, width, label='After')

plt.xticks(x, ['T1', 'T3'])
plt.ylabel('Passengers Handled')
plt.title('Before vs After Reassignment - Passenger Load')
plt.legend()
plt.tight_layout()
plt.show()


# ------------------ BEFORE vs AFTER IMPACT SUMMARY TABLE (PRINT) ------------------
print("\n=== SUMMARY: BEFORE vs AFTER ===")
print(f"T1 Passengers: Before={pax_T1_before:,} → After={pax_T1_after:,}")
print(f"T3 Passengers: Before={pax_T3_before:,} → After={pax_T3_after:,}")
print(f"T1 Wait: Before={BASE_WAITS['T1']:.2f} → After={after_T1:.2f}")
print(f"T3 Wait: Before={BASE_WAITS['T3']:.2f} → After={after_T3:.2f}")