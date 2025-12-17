
import simpy
import numpy as np
import pandas as pd
from statistics import mean

# ------------------- PARAMETERS -------------------
RANDOM_SEED = 42
RUN_TIME = 1440  # minutes (24 hours)
P_DIGI = 0.19

np.random.seed(RANDOM_SEED)

# ------------------- TIME INTERVALS -------------------
time_blocks = np.array([
    0, 60, 120, 180, 240, 300, 315, 330, 345, 360, 375, 390, 405,
    420, 435, 450, 465, 480, 495, 510, 525, 540, 600, 660, 720, 780,
    840, 900, 960, 975, 990, 1005, 1020, 1035, 1050, 1065, 1080,
    1095, 1110, 1125, 1140, 1155, 1170, 1185, 1200, 1260, 1320, 1380
])

# ------------------- SERVICE TIME PROFILES -------------------
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

# ------------------- TIME-DEPENDENT SAMPLERS -------------------
def get_service_time(arr, t):
    idx = np.searchsorted(time_blocks, t, side="right") - 1
    idx = max(0, min(idx, len(arr) - 1))
    return np.random.exponential(arr[idx])

def D_gate_Digi_T2(): return 0.5 + np.random.exponential(0.05)
def D_gate_Reg_T2(t): return get_service_time(service_gate, t)
def D_checkin_T2(t):  return get_service_time(service_checkin, t)
def D_security_T2(t): return get_service_time(service_security, t)
def D_walking():      return np.random.triangular(5.0, 10.0, 10.0)
def D_waiting():      return np.random.uniform(20.0, 40.0)
def D_boarding():     return np.random.triangular(5.0, 8.0, 10.0)

# ------------------- LOAD FLIGHT DATA -------------------
df = pd.read_csv('data/Terminal2 Flight Data.csv')  # uncomment if needed

# ==========================================================
# ✅ FUNCTION: RUN SIMULATION FOR TERMINAL 2 (IndiGo + Akasa Air)
# ==========================================================
def run_simulation_T2(cap_gate_reg, cap_gate_digi,
                      cap_checkin_indigo, cap_checkin_akasa,
                      cap_security, num_gates, cap_boarding_per_gate):
    """
    Expects a global `df` containing Terminal 2 schedule with columns:
    ['Dep_time_min', 'N_passengers', 'Airline'] where Airline in {'IndiGo','Akasa Air'}
    """

    global cycle_times, waits, queue_lengths, busy_time, throughput, results_by_airline, system_count

    # Reset metrics
    cycle_times = []
    waits = {"gate_reg": [], "gate_digi": [], "checkin_indigo": [], "checkin_akasa_air": [],
             "security": [], "boarding": [], "waiting_hall": []}
    queue_lengths = {"checkin": [], "security": [], "boarding": []}
    busy_time = {"checkin": 0, "security": 0, "boarding": 0}
    throughput = np.zeros(28)
    results_by_airline = {"IndiGo": [], "Akasa Air": []}
    system_count = []

    # -------- Passenger --------
    def passenger(env, pid, airline, gate_reg, gate_digi, checkin_indigo, checkin_akasa, securities, boardings):
        t0 = env.now
        system_count.append((env.now, +1))

        # GATE SELECTION
        if np.random.rand() < P_DIGI:
            chosen = min(gate_digi, key=lambda r: len(r.queue))
            with chosen.request() as req:
                t_req = env.now; yield req
                waits["gate_digi"].append(env.now - t_req)
                yield env.timeout(D_gate_Digi_T2())
        else:
            chosen = min(gate_reg, key=lambda r: len(r.queue))
            with chosen.request() as req:
                t_req = env.now; yield req
                waits["gate_reg"].append(env.now - t_req)
                yield env.timeout(D_gate_Reg_T2(env.now))

        # CHECK-IN
        if airline.lower() == "indigo":
            chosen = min(checkin_indigo, key=lambda r: len(r.queue))
            queue_lengths["checkin"].append(len(chosen.queue))
            with chosen.request() as req:
                t_req = env.now; yield req
                waits["checkin_indigo"].append(env.now - t_req)
                st = D_checkin_T2(env.now); yield env.timeout(st)
                busy_time["checkin"] += st
        else:  # Akasa Air
            chosen = min(checkin_akasa, key=lambda r: len(r.queue))
            queue_lengths["checkin"].append(len(chosen.queue))
            with chosen.request() as req:
                t_req = env.now; yield req
                waits["checkin_akasa_air"].append(env.now - t_req)
                st = D_checkin_T2(env.now); yield env.timeout(st)
                busy_time["checkin"] += st

        # SECURITY
        chosen = min(securities, key=lambda r: len(r.queue))
        queue_lengths["security"].append(len(chosen.queue))
        with chosen.request() as req:
            t_req = env.now; yield req
            waits["security"].append(env.now - t_req)
            st = D_security_T2(env.now); yield env.timeout(st)
            busy_time["security"] += st

        # WAITING AREA
        waiting_time = D_waiting()
        waits["waiting_hall"].append(waiting_time)
        yield env.timeout(D_walking() + waiting_time)

        # BOARDING — uses num_gates and per-gate capacity
        chosen = min(boardings, key=lambda r: len(r.queue))
        queue_lengths["boarding"].append(len(chosen.queue))
        with chosen.request() as req:
            t_req = env.now; yield req
            waits["boarding"].append(env.now - t_req)
            st = D_boarding(); yield env.timeout(st)
            busy_time["boarding"] += st

        cycle_times.append(env.now - t0)
        results_by_airline[airline].append(env.now - t0)
        system_count.append((env.now, -1))

        # hourly completion count
        idx = min(int(env.now // 60), len(throughput) - 1)
        throughput[idx] += 1

    # -------- Flight generator --------
    def flight_source(env, gate_reg, gate_digi, checkin_indigo, checkin_akasa, securities, boardings):
        pid = 0
        for _, row in df.iterrows():
            dep_time = row["Dep_time_min"]
            n_passengers = int(row["N_passengers"])
            airline = row["Airline"]

            # arrivals between dep-120 and dep-60 (clipped to [0, RUN_TIME])
            arr_start = max(0.0, dep_time - 120.0)
            arr_end   = max(arr_start, min(RUN_TIME, dep_time - 60.0))

            if n_passengers > 0:
                gap = (arr_end - arr_start) / n_passengers if (arr_end - arr_start) > 0 else 0
                if gap <= 0:
                    arrivals = np.full(n_passengers, arr_start)
                else:
                    arrivals = arr_start + np.cumsum(np.random.exponential(gap, n_passengers))
                arrivals = np.clip(arrivals, arr_start, arr_end)
                arrivals.sort()

                for arr_time in arrivals:
                    yield env.timeout(max(0, arr_time - env.now))
                    pid += 1
                    env.process(passenger(env, pid, airline, gate_reg, gate_digi,
                                          checkin_indigo, checkin_akasa, securities, boardings))

    # -------- Environment + Resources --------
    env = simpy.Environment()
    gate_reg        = [simpy.Resource(env, capacity=1)                 for _ in range(cap_gate_reg)]
    gate_digi       = [simpy.Resource(env, capacity=1)                 for _ in range(cap_gate_digi)]
    checkin_indigo  = [simpy.Resource(env, capacity=1)                 for _ in range(cap_checkin_indigo)]
    checkin_akasa   = [simpy.Resource(env, capacity=1)                 for _ in range(cap_checkin_akasa)]
    securities      = [simpy.Resource(env, capacity=1)                 for _ in range(cap_security)]
    boardings       = [simpy.Resource(env, capacity=cap_boarding_per_gate) for _ in range(num_gates)]

    env.process(flight_source(env, gate_reg, gate_digi, checkin_indigo, checkin_akasa, securities, boardings))
    env.run(until=RUN_TIME + 180)

    def avg(lst): return mean(lst) if lst else 0.0

    return {
        # --- Wait Time Metrics ---
        "total_avg_wait": avg(waits["gate_reg"] + waits["gate_digi"])
                          + avg(waits["checkin_indigo"] + waits["checkin_akasa_air"])
                          + avg(waits["security"])
                          + avg(waits["boarding"]),

        "avg_wait_gate":       avg(waits["gate_reg"] + waits["gate_digi"]),
        "avg_wait_checkin":    avg(waits["checkin_indigo"] + waits["checkin_akasa_air"]),
        "avg_wait_security":   avg(waits["security"]),
        "avg_wait_boarding":   avg(waits["boarding"]),

        # --- Queue Length Metrics ---
        "avg_q_checkin":  np.mean(queue_lengths["checkin"]) if queue_lengths["checkin"] else 0.0,
        "avg_q_security": np.mean(queue_lengths["security"]) if queue_lengths["security"] else 0.0,
        "avg_q_boarding": np.mean(queue_lengths["boarding"]) if queue_lengths["boarding"] else 0.0,

        # --- System Performance ---
        "avg_time_in_system": np.mean(cycle_times) if len(cycle_times) else 0.0,
        "passengers_completed": len(cycle_times),

        "utilization": {k: busy_time[k] / RUN_TIME for k in busy_time},

        # (optional extras you may want to use downstream)
        "throughput_per_hour": throughput.copy(),
        "results_by_airline": {k: np.mean(v) if v else 0.0 for k, v in results_by_airline.items()}
    }

# df = pd.read_csv('data/Terminal2 Flight Data.csv')

run_simulation_T2(
    cap_gate_reg=24,
    cap_gate_digi=4,
    cap_checkin_indigo=28,
    cap_checkin_akasa=5,
    cap_security=55,
    num_gates=7,
    cap_boarding_per_gate=100
)

import matplotlib
matplotlib.use("Agg")  # disables live display; you can still save figures

# pip install tqdm

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
from IPython.display import clear_output

# ============================
# TERMINAL-2 CAPACITY OPTIMIZER (Colab Live + Excel + Sweeps)
# ============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output

plt.rcParams["figure.dpi"] = 110

# -------------------- TUNING KNOBS --------------------
N_REP = 3
ABS_IMPROVE_MIN = 0.5
REL_IMPROVE_MIN = 0.02

# We optimize these, in this order
RESOURCES_ORDER_T2 = [
    "checkin_indigo", "checkin_akasa", "security",
    "gate_reg", "gate_digi", "num_gates"
]

# Safety caps
MAX_CAPS_T2 = {
    "gate_reg": 80,
    "gate_digi": 40,
    "checkin_indigo": 120,
    "checkin_akasa": 60,
    "security": 160,
    "num_gates": 40
}

# Fixed per-gate boarding capacity
CAP_BOARDING_PER_GATE_T2 = 200

trend_T2 = []


# -------------------- FORMATTING & BOTTLENECK --------------------
def _format_caps_T2(caps, include_board_cap=True):
    base = (f"GReg={caps['gate_reg']}, GDigi={caps['gate_digi']}, "
            f"ChkInd={caps['checkin_indigo']}, ChkAkasa={caps['checkin_akasa']}, "
            f"Sec={caps['security']}, Gates={caps['num_gates']}")
    if include_board_cap:
        base += f", GateCap={CAP_BOARDING_PER_GATE_T2}"
    return base

def _bottleneck_from_waits_T2(avg_waits):
    gate = avg_waits["gate"]
    checkin = avg_waits["checkin"]
    security = avg_waits["security"]
    boarding = avg_waits["boarding"]
    total = max(1e-9, gate + checkin + security + boarding)
    shares = {
        "Gate": gate/total,
        "Check-in": checkin/total,
        "Security": security/total,
        "Boarding": boarding/total
    }
    bottleneck = max(shares.items(), key=lambda kv: kv[1])[0]
    return bottleneck, shares


# -------------------- LIVE DASHBOARD --------------------
def _live_dashboard_T2(stats, iteration):
    clear_output(wait=True)
    waits = stats["avg_waits"]
    bn, shares = _bottleneck_from_waits_T2(waits)

    plt.figure(figsize=(12,4))

    # (1) Stage waits
    plt.subplot(1,3,1)
    plt.bar(["Gate","Check-in","Security","Boarding"],
            [waits["gate"], waits["checkin"], waits["security"], waits["boarding"]],
            color=['#4e79a7','#f28e2b','#e15759','#59a14f'])
    plt.title("Stage Waiting Times")
    plt.ylabel("Minutes")

    # (2) Mix %
    plt.subplot(1,3,2)
    sizes = [waits["gate"], waits["checkin"], waits["security"], waits["boarding"]]
    plt.pie(sizes, labels=["Gate","Check-in","Security","Boarding"], autopct="%1.1f%%")
    plt.title(f"Bottleneck Mix (Bn: {bn})")

    # (3) Trend
    trend_T2.append(waits["total"])
    plt.subplot(1,3,3)
    plt.plot(trend_T2, '-o', linewidth=2)
    plt.title("Total Avg Wait Across Accepted Steps")
    plt.xlabel("Accepted step #")
    plt.ylabel("Minutes")
    plt.grid(alpha=0.3)

    plt.suptitle(f"Optimization Progress – Iteration {iteration}", y=1.03)
    plt.tight_layout()
    plt.show()


# -------------------- AVERAGED RUN WRAPPER --------------------
def _run_avg_T2(caps, n_rep=N_REP, base_seed=12345):
    results = []
    seeds = np.random.SeedSequence(base_seed).spawn(n_rep)
    for i, s in enumerate(seeds):
        np.random.seed(int(s.entropy))
        res = run_simulation_T2(
            caps["gate_reg"],
            caps["gate_digi"],
            caps["checkin_indigo"],
            caps["checkin_akasa"],
            caps["security"],
            caps["num_gates"],
            CAP_BOARDING_PER_GATE_T2
        )
        results.append(res)

    def A(key): return float(np.mean([r[key] for r in results]))

    avg_waits = {
        "gate": A("avg_wait_gate"),
        "checkin": A("avg_wait_checkin"),
        "security": A("avg_wait_security"),
        "boarding": A("avg_wait_boarding"),
        "total": A("total_avg_wait"),
    }
    avg_cycle = A("avg_time_in_system")
    completed = int(np.mean([r["passengers_completed"] for r in results]))
    util = {k: float(np.mean([r["utilization"][k] for r in results])) for k in results[0]["utilization"]}

    bn, shares = _bottleneck_from_waits_T2(avg_waits)
    print(f"   ↳ Bottleneck: {bn.upper()}")

    return {
        "avg_waits": avg_waits,
        "avg_cycle": avg_cycle,
        "completed": completed,
        "utilization": util,
        "bottleneck": bn,
        "shares": shares
    }


# -------------------- SAVE TO EXCEL --------------------
def save_history_T2(history_rows, filename="Optimization_Results_T2.xlsx"):
    df_hist = pd.DataFrame(history_rows)
    summary_candidates = df_hist[df_hist["event"].isin(["accept","final","baseline"])]
    last_row = summary_candidates.iloc[-1].to_dict() if len(summary_candidates) else df_hist.iloc[-1].to_dict()
    df_sum = pd.DataFrame([last_row])
    with pd.ExcelWriter(filename, engine="xlsxwriter") as w:
        df_hist.to_excel(w, index=False, sheet_name="history")
        df_sum.to_excel(w, index=False, sheet_name="summary")
    print(f"📄 Saved optimization history & summary to: {filename}")


# -------------------- GREEDY PLATEAU OPTIMIZER --------------------
def optimize_capacities_T2(
    start_caps=dict(gate_reg=24, gate_digi=4, checkin_indigo=28, checkin_akasa=5, security=55, num_gates=7),
    abs_improve=ABS_IMPROVE_MIN,
    rel_improve=REL_IMPROVE_MIN,
    resources_order=RESOURCES_ORDER_T2,
    n_rep=N_REP,
    verbose=True,
    save_excel=True,
    excel_filename="Optimization_Results_T2.xlsx"
):

    def _cap_ok(res, val): return val <= MAX_CAPS_T2.get(res, val)
    caps = start_caps.copy()
    history_rows = []
    iteration = 0

    # ---------- BASELINE ----------
    base_stats = _run_avg_T2(caps, n_rep=n_rep)
    best_stats = base_stats
    best_obj = base_stats["avg_waits"]["total"]

    print(f"[Baseline] {_format_caps_T2(caps)} -> TotalAvgWait={best_obj:.3f} min")

    history_rows.append({
        "event": "baseline",
        **caps,
        "cap_per_gate": CAP_BOARDING_PER_GATE_T2,
        "total_avg_wait": best_obj,
        "gate": base_stats["avg_waits"]["gate"],
        "checkin": base_stats["avg_waits"]["checkin"],
        "security": base_stats["avg_waits"]["security"],
        "boarding": base_stats["avg_waits"]["boarding"],
        **{f"util_{k}": v for k,v in base_stats["utilization"].items()}
    })

    _live_dashboard_T2(base_stats, iteration)

    # ---------- IMPROVEMENT LOOP ----------
    improved_any = True
    while improved_any:
        improved_any = False
        for res in tqdm(resources_order, desc="Resources pass (T2)"):
            local_improved = False
            while True:
                new_cap = caps[res] + 1
                if not _cap_ok(res, new_cap):
                    break

                trial_caps = caps.copy()
                trial_caps[res] = new_cap
                trial_stats = _run_avg_T2(trial_caps, n_rep=n_rep)

                trial_obj = trial_stats["avg_waits"]["total"]
                abs_gain = best_obj - trial_obj
                rel_gain = abs_gain / best_obj if best_obj > 0 else 0.0

                _live_dashboard_T2(trial_stats, iteration + 0.1)

                if (abs_gain >= abs_improve) or (rel_gain >= rel_improve):
                    caps = trial_caps
                    best_stats = trial_stats
                    best_obj = trial_obj
                    iteration += 1
                    local_improved = True
                    improved_any = True

                    print(f"[Improve] {res:>16} -> {caps[res]:>3} | TotalAvgWait={best_obj:.3f}")

                    history_rows.append({
                        "event": "accept",
                        **caps,
                        "cap_per_gate": CAP_BOARDING_PER_GATE_T2,
                        "total_avg_wait": best_obj,
                        "gate": best_stats["avg_waits"]["gate"],
                        "checkin": best_stats["avg_waits"]["checkin"],
                        "security": best_stats["avg_waits"]["security"],
                        "boarding": best_stats["avg_waits"]["boarding"],
                        **{f"util_{k}": v for k,v in best_stats["utilization"].items()}
                    })

                    _live_dashboard_T2(best_stats, iteration)

                else:
                    break

            if verbose and not local_improved:
                print(f"[Plateau] {res:>16} at {caps[res]}")

    # ---------- FINAL STATE ----------
    history_rows.append({
        "event": "final",
        **caps,
        "cap_per_gate": CAP_BOARDING_PER_GATE_T2,
        "total_avg_wait": best_obj,
        "gate": best_stats["avg_waits"]["gate"],
        "checkin": best_stats["avg_waits"]["checkin"],
        "security": best_stats["avg_waits"]["security"],
        "boarding": best_stats["avg_waits"]["boarding"],
        **{f"util_{k}": v for k,v in best_stats["utilization"].items()}
    })

    if save_excel:
        save_history_T2(history_rows, filename=excel_filename)

    return caps, best_stats, history_rows, base_stats

# ============================
# TERMINAL-2 CAPACITY OPTIMIZER — MONOTONIC (Only Increase, Never Decrease)
# ============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output

plt.rcParams["figure.dpi"] = 110

# -------------------- TUNING KNOBS --------------------
N_REP = 3
ABS_IMPROVE_MIN = 0.5     # minimal absolute improvement (minutes)
REL_IMPROVE_MIN = 0.02    # minimal relative improvement (fraction)

# We optimize these, in this order (T2 airlines: IndiGo + Akasa)
RESOURCES_ORDER_T2 = [
    "checkin_indigo", "checkin_akasa", "security",
    "gate_reg", "gate_digi", "num_gates"
]

# Safety caps
MAX_CAPS_T2 = {
    "gate_reg": 80,
    "gate_digi": 40,
    "checkin_indigo": 120,
    "checkin_akasa": 60,
    "security": 160,
    "num_gates": 40
}

# Fixed per-gate boarding capacity (not optimized here)
CAP_BOARDING_PER_GATE_T2 = 200

trend_T2 = []  # live trend for dashboard


# -------------------- HELPERS --------------------
def _format_caps_T2(caps, include_board_cap=True):
    base = (f"GReg={caps['gate_reg']}, GDigi={caps['gate_digi']}, "
            f"ChkInd={caps['checkin_indigo']}, ChkAkasa={caps['checkin_akasa']}, "
            f"Sec={caps['security']}, Gates={caps['num_gates']}")
    if include_board_cap:
        base += f", GateCap={CAP_BOARDING_PER_GATE_T2}"
    return base

def _bottleneck_from_waits_T2(avg_waits):
    gate = avg_waits["gate"]
    checkin = avg_waits["checkin"]
    security = avg_waits["security"]
    boarding = avg_waits["boarding"]
    total = max(1e-9, gate + checkin + security + boarding)
    shares = {"Gate": gate/total, "Check-in": checkin/total,
              "Security": security/total, "Boarding": boarding/total}
    bottleneck = max(shares.items(), key=lambda kv: kv[1])[0]
    return bottleneck, shares

def _live_dashboard_T2(stats, iteration):
    clear_output(wait=True)
    waits = stats["avg_waits"]
    bn, shares = _bottleneck_from_waits_T2(waits)

    plt.figure(figsize=(12,4))

    # (1) Stage waits
    plt.subplot(1,3,1)
    plt.bar(["Gate","Check-in","Security","Boarding"],
            [waits["gate"], waits["checkin"], waits["security"], waits["boarding"]],
            color=['#4e79a7','#f28e2b','#e15759','#59a14f'])
    plt.title("Stage Waiting Times")
    plt.ylabel("Minutes")

    # (2) Mix %
    plt.subplot(1,3,2)
    sizes = [waits["gate"], waits["checkin"], waits["security"], waits["boarding"]]
    plt.pie(sizes, labels=["Gate","Check-in","Security","Boarding"], autopct="%1.1f%%")
    plt.title(f"Bottleneck Mix (Bn: {bn})")

    # (3) Trend
    trend_T2.append(waits["total"])
    plt.subplot(1,3,3)
    plt.plot(trend_T2, '-o', linewidth=2)
    plt.title("Total Avg Wait Across Accepted Steps")
    plt.xlabel("Accepted step #")
    plt.ylabel("Minutes")
    plt.grid(alpha=0.3)

    plt.suptitle(f"Optimization Progress – Iteration {iteration}", y=1.03)
    plt.tight_layout()
    plt.show()


# -------------------- AVERAGED RUN WRAPPER --------------------
def _run_avg_T2(caps, n_rep=N_REP, base_seed=12345):
    """
    Requires your run_simulation_T2(...) with signature:
      run_simulation_T2(gate_reg, gate_digi, checkin_indigo, checkin_akasa, security, num_gates, cap_boarding_per_gate)
    """
    results = []
    seeds = np.random.SeedSequence(base_seed).spawn(n_rep)
    for s in seeds:
        np.random.seed(int(s.entropy))
        res = run_simulation_T2(
            caps["gate_reg"],
            caps["gate_digi"],
            caps["checkin_indigo"],
            caps["checkin_akasa"],
            caps["security"],
            caps["num_gates"],
            CAP_BOARDING_PER_GATE_T2
        )
        results.append(res)

    def A(key): return float(np.mean([r[key] for r in results]))

    avg_waits = {
        "gate": A("avg_wait_gate"),
        "checkin": A("avg_wait_checkin"),
        "security": A("avg_wait_security"),
        "boarding": A("avg_wait_boarding"),
        "total": A("total_avg_wait"),
    }
    avg_cycle = A("avg_time_in_system")
    completed = int(np.mean([r["passengers_completed"] for r in results]))
    util_keys = list(results[0]["utilization"].keys())
    utilization = {k: float(np.mean([r["utilization"][k] for r in results])) for k in util_keys}

    return {
        "avg_waits": avg_waits,
        "avg_cycle": avg_cycle,
        "completed": completed,
        "utilization": utilization
    }


# -------------------- SAVE TO EXCEL --------------------
def save_history_T2(history_rows, filename="Optimization_Results_T2.xlsx"):
    df_hist = pd.DataFrame(history_rows)
    summary_candidates = df_hist[df_hist["event"].isin(["accept","final","baseline"])]
    last_row = summary_candidates.iloc[-1].to_dict() if len(summary_candidates) else df_hist.iloc[-1].to_dict()
    df_sum = pd.DataFrame([last_row])
    with pd.ExcelWriter(filename, engine="xlsxwriter") as w:
        df_hist.to_excel(w, index=False, sheet_name="history")
        df_sum.to_excel(w, index=False, sheet_name="summary")
    print(f"📄 Saved optimization history & summary to: {filename}")


# -------------------- GREEDY PLATEAU OPTIMIZER (MONOTONIC) --------------------
def optimize_capacities_T2(
    start_caps=dict(gate_reg=24, gate_digi=4, checkin_indigo=28, checkin_akasa=5, security=55, num_gates=7),
    abs_improve=ABS_IMPROVE_MIN,
    rel_improve=REL_IMPROVE_MIN,
    resources_order=RESOURCES_ORDER_T2,
    n_rep=N_REP,
    verbose=True,
    save_excel=True,
    excel_filename="Optimization_Results_T2.xlsx",
    show_live=True
):
    """
    Monotonic optimizer:
      • Only +1 increments are attempted and accepted.
      • Final capacities are GUARANTEED >= start_caps for every resource.
      • If a +1 doesn't pass the improvement threshold, that resource stops (no decrements).
    """
    # Immutable lower bounds = given start
    START = start_caps.copy()

    # Local guard: clamp any accidental decrease back to START
    def _clamp_to_start(caps):
        for k in START:
            if caps[k] < START[k]:
                caps[k] = START[k]

    # Initialize
    caps = start_caps.copy()
    _clamp_to_start(caps)

    history_rows = []
    iteration = 0

    # ---------- BASELINE ----------
    base_stats = _run_avg_T2(caps, n_rep=n_rep)
    best_stats = base_stats
    best_obj = base_stats["avg_waits"]["total"]

    if verbose:
        print(f"[Baseline] {_format_caps_T2(caps)} -> TotalAvgWait={best_obj:.3f} min")

    # Log baseline with explicit "start_*" fields for plotting safety
    history_rows.append({
        "event": "baseline",
        **caps,
        **{f"start_{k}": START[k] for k in START},
        "cap_per_gate": CAP_BOARDING_PER_GATE_T2,
        "total_avg_wait": best_obj,
        "gate": base_stats["avg_waits"]["gate"],
        "checkin": base_stats["avg_waits"]["checkin"],
        "security": base_stats["avg_waits"]["security"],
        "boarding": base_stats["avg_waits"]["boarding"],
        **{f"util_{k}": v for k, v in base_stats["utilization"].items()}
    })

    if show_live:
        _live_dashboard_T2(base_stats, iteration)

    # ---------- IMPROVEMENT LOOP ----------
    improved_any = True
    while improved_any:
        improved_any = False
        for res in tqdm(resources_order, desc="Resources pass (T2)"):
            local_improved = False

            while True:
                new_cap = caps[res] + 1
                if new_cap > MAX_CAPS_T2[res]:
                    if verbose:
                        print(f"[Limit] {res} cannot exceed {MAX_CAPS_T2[res]}")
                    break

                trial_caps = caps.copy()
                trial_caps[res] = new_cap
                _clamp_to_start(trial_caps)

                trial_stats = _run_avg_T2(trial_caps, n_rep=n_rep)
                trial_obj = trial_stats["avg_waits"]["total"]
                abs_gain = best_obj - trial_obj
                rel_gain = abs_gain / best_obj if best_obj > 0 else 0.0

                if show_live:
                    _live_dashboard_T2(trial_stats, iteration + 0.1)

                # Accept ONLY +1 improvements; never revert
                if (abs_gain >= abs_improve) or (rel_gain >= rel_improve):
                    caps = trial_caps
                    best_stats = trial_stats
                    best_obj = trial_obj
                    iteration += 1
                    local_improved = True
                    improved_any = True

                    if verbose:
                        print(f"[Improve] {res:>16} -> {caps[res]:>3} | TotalAvgWait={best_obj:.3f} "
                              f"(abs_gain={abs_gain:.3f}, rel={rel_gain*100:.2f}%)")

                    history_rows.append({
                        "event": "accept",
                        **caps,
                        **{f"start_{k}": START[k] for k in START},
                        "cap_per_gate": CAP_BOARDING_PER_GATE_T2,
                        "total_avg_wait": best_obj,
                        "gate": best_stats["avg_waits"]["gate"],
                        "checkin": best_stats["avg_waits"]["checkin"],
                        "security": best_stats["avg_waits"]["security"],
                        "boarding": best_stats["avg_waits"]["boarding"],
                        **{f"util_{k}": v for k, v in best_stats["utilization"].items()}
                    })

                    if show_live:
                        _live_dashboard_T2(best_stats, iteration)
                else:
                    # stop trying further +1 for this resource
                    if verbose:
                        print(f"[Plateau] {res:>16} stays at {caps[res]}")
                    break

            _clamp_to_start(caps)  # extra safety

    # ---------- FINAL STATE ----------
    _clamp_to_start(caps)
    # Hard assertion: NEVER lower than start
    for k in START:
        assert caps[k] >= START[k], f"Invariant violated: {k} decreased ({caps[k]} < {START[k]})"

    history_rows.append({
        "event": "final",
        **caps,
        **{f"start_{k}": START[k] for k in START},
        "cap_per_gate": CAP_BOARDING_PER_GATE_T2,
        "total_avg_wait": best_obj,
        "gate": best_stats["avg_waits"]["gate"],
        "checkin": best_stats["avg_waits"]["checkin"],
        "security": best_stats["avg_waits"]["security"],
        "boarding": best_stats["avg_waits"]["boarding"],
        **{f"util_{k}": v for k, v in best_stats["utilization"].items()}
    })

    if save_excel:
        save_history_T2(history_rows, filename=excel_filename)

    # Also return the explicit START caps so plots can use them directly
    return caps, best_stats, history_rows, base_stats, START

start_caps_T2 = dict(
    gate_reg=24,
    gate_digi=4,
    checkin_indigo=28,
    checkin_akasa=5,
    security=55,
    num_gates=7
)

best_caps_T2, best_stats_T2, history_rows_T2, baseline_stats_T2, START_T2 = optimize_capacities_T2(
    start_caps=start_caps_T2,
    save_excel=True,
    excel_filename="Optimization_Results_T2.xlsx",
    show_live=False   # set True if you want the live dashboard
)

# --- Safe plot: counters before (START) vs after (best) ---
import numpy as np
resources = ["Gate (Reg)", "Gate (Digi)", "Check-in (Indigo)", "Check-in (Akasa)", "Security Lanes", "Boarding Gates"]
before_caps = [START_T2["gate_reg"], START_T2["gate_digi"], START_T2["checkin_indigo"],
               START_T2["checkin_akasa"], START_T2["security"], START_T2["num_gates"]]
after_caps  = [best_caps_T2["gate_reg"], best_caps_T2["gate_digi"], best_caps_T2["checkin_indigo"],
               best_caps_T2["checkin_akasa"], best_caps_T2["security"], best_caps_T2["num_gates"]]

x = np.arange(len(resources)); w = 0.35
plt.figure(figsize=(10,5))
b1 = plt.bar(x-w/2, before_caps, w, label="Before", color="#4C72B0", edgecolor="black")
b2 = plt.bar(x+w/2, after_caps,  w, label="After",  color="#DD8452", edgecolor="black")
for b in list(b1)+list(b2):
    plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.4, f"{int(b.get_height())}", ha='center', fontsize=10)
plt.xticks(x, resources, rotation=15); plt.ylabel("Number of Counters / Gates")
plt.title("T2 – Resource Counters Before vs After Optimization"); plt.grid(axis="y", linestyle="--", alpha=0.35); plt.legend()
plt.tight_layout(); plt.show()

# pip install xlsxwriter

# Commented out IPython magic to ensure Python compatibility.
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
# %matplotlib inline
plt.rcParams["figure.dpi"] = 120

# ============================
# T2 — FINAL 5 PLOTS (Before vs After)
# ============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Resolve objects from your session ----------
# baseline stats (from optimizer return)
try:
    _baseline_stats = baseline_stats_T2    # preferred name in your calls
except NameError:
    _baseline_stats = base_stats_T2        # fallback name if used earlier

# starting capacities (hard "before" truth for counters)
try:
    START_T2  # if returned by optimizer
    _start_caps = START_T2
except NameError:
    _start_caps = start_caps_T2  # else use what you passed in

# history rows and best (final) stats
df_T2 = pd.DataFrame(history_rows_T2)

# last baseline + final rows from history (for waits/util/util history)
baseline_row_T2 = df_T2[df_T2.event=="baseline"].iloc[-1].copy()
final_row_T2    = df_T2[df_T2.event=="final"].iloc[-1].copy()

# ---------- 1) Stage Waiting Times (Before vs After) ----------
labels = ["Gate", "Check-in", "Security", "Boarding"]
before_waits = [
    _baseline_stats["avg_waits"]["gate"],
    _baseline_stats["avg_waits"]["checkin"],
    _baseline_stats["avg_waits"]["security"],
    _baseline_stats["avg_waits"]["boarding"],
]
after_waits = [
    best_stats_T2["avg_waits"]["gate"],
    best_stats_T2["avg_waits"]["checkin"],
    best_stats_T2["avg_waits"]["security"],
    best_stats_T2["avg_waits"]["boarding"],
]

x = np.arange(len(labels)); width = 0.38
plt.figure(figsize=(8,5))
plt.bar(x - width/2, before_waits, width, color="#4C72B0", edgecolor="black", label="Before")
plt.bar(x + width/2, after_waits,  width, color="#DD8452", edgecolor="black", label="After")
plt.xticks(x, labels, fontsize=12)
plt.ylabel("Average Waiting Time (min)", fontsize=12)
plt.title("T2 – Stage-wise Waiting Times (Before vs After)", fontsize=14, weight="bold")
plt.legend(); plt.grid(axis="y", linestyle="--", alpha=0.35)
plt.tight_layout(); plt.show()

# ---------- 2) Bottleneck Share (Pies) ----------
before_share = np.array(before_waits) / max(1e-9, np.sum(before_waits))
after_share  = np.array(after_waits)  / max(1e-9, np.sum(after_waits))

explode_before = [0.08 if v == before_share.max() else 0 for v in before_share]
explode_after  = [0.08 if v == after_share.max()  else 0 for v in after_share]
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

fig, ax = plt.subplots(1, 2, figsize=(10,5))
ax[0].pie(before_share, labels=labels, autopct="%1.1f%%", explode=explode_before,
          startangle=90, colors=colors, shadow=True)
ax[0].set_title("Before Optimization (T2)", fontsize=13)
ax[1].pie(after_share, labels=labels, autopct="%1.1f%%", explode=explode_after,
          startangle=90, colors=colors, shadow=True)
ax[1].set_title("After Optimization (T2)", fontsize=13)
plt.suptitle("T2 – Bottleneck Stage Share Comparison", fontsize=15, weight='bold')
plt.tight_layout(); plt.show()

# ---------- 3) Resource Counters (Before vs After, Only-Increase Safe) ----------
resources = ["Gate (Reg)", "Gate (Digi)", "Check-in (Indigo)", "Check-in (Akasa)", "Security Lanes", "Boarding Gates"]
before_caps = [
    _start_caps["gate_reg"], _start_caps["gate_digi"],
    _start_caps["checkin_indigo"], _start_caps["checkin_akasa"],
    _start_caps["security"], _start_caps["num_gates"]
]
after_caps = [
    final_row_T2["gate_reg"], final_row_T2["gate_digi"],
    final_row_T2["checkin_indigo"], final_row_T2["checkin_akasa"],
    final_row_T2["security"], final_row_T2["num_gates"]
]

# guard (should already hold true, but never hurts)
after_caps = [max(a, b) for a, b in zip(after_caps, before_caps)]

x = np.arange(len(resources)); width = 0.35
fig, ax = plt.subplots(figsize=(10,5))
bars1 = ax.bar(x - width/2, before_caps, width, label="Before", color="#4C72B0", edgecolor="black")
bars2 = ax.bar(x + width/2, after_caps,  width, label="After",  color="#DD8452", edgecolor="black")
for b in list(bars1) + list(bars2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.4, f"{int(b.get_height())}", ha='center', fontsize=10)
ax.set_ylabel("Number of Counters / Gates", fontsize=12)
ax.set_title("T2 – Resource Counters Before vs After Optimization", fontsize=14, weight='bold')
ax.set_xticks(x); ax.set_xticklabels(resources, rotation=15)
ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.35)
plt.tight_layout(); plt.show()

# ---------- 4) Utilization Comparison ----------
# keep a stable order if available
util_order = ["checkin", "security", "boarding"]
u_keys = [k for k in util_order if k in best_stats_T2["utilization"]]
# if unknown order, fall back to whatever is present
if not u_keys:
    u_keys = list(best_stats_T2["utilization"].keys())

util_before = [_baseline_stats["utilization"][k] for k in u_keys]
util_after  = [best_stats_T2["utilization"][k]    for k in u_keys]

x = np.arange(len(u_keys)); width = 0.35
fig, ax = plt.subplots(figsize=(8,5))
bars1 = ax.bar(x - width/2, util_before, width, label="Before", color="#4C72B0", edgecolor="black")
bars2 = ax.bar(x + width/2, util_after,  width, label="After",  color="#DD8452", edgecolor="black")
for b in list(bars1) + list(bars2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"{b.get_height():.2f}", ha='center', fontsize=10)
ax.set_ylabel("Utilization (busy fraction)", fontsize=12)
ax.set_title("T2 – Utilization Comparison (Before vs After)", fontsize=14, weight='bold')
ax.set_xticks(x); ax.set_xticklabels([k.capitalize() for k in u_keys], fontsize=11)
ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.35)
plt.tight_layout(); plt.show()

# ---------- 5) Optimization Trend (Accepted steps only) ----------
df_acc_T2 = df_T2[df_T2.event=="accept"]
plt.figure(figsize=(7,4))
if len(df_acc_T2):
    plt.plot(df_acc_T2.total_avg_wait.values, "-o", color="#8A4FFF", linewidth=2)
else:
    # show baseline->final if no accepts were logged
    plt.plot([_baseline_stats["avg_waits"]["total"], best_stats_T2["avg_waits"]["total"]],
             "-o", color="#8A4FFF", linewidth=2)
plt.xlabel("Accepted Improvement Step")
plt.ylabel("Total Avg Wait (min)")
plt.title("T2 – Optimization Convergence Trend", fontsize=14, weight="bold")
plt.grid(alpha=0.4); plt.tight_layout(); plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Convert history to DataFrame
df_T2 = pd.DataFrame(history_rows_T2)

# Get baseline and final states
baseline_row_T2 = df_T2[df_T2.event=="baseline"].iloc[-1].copy()

# Force baseline to match your actual start values
baseline_row_T2["security"] = start_caps_T2["security"]

final_T2    = df_T2[df_T2.event=="final"].iloc[-1]

# Resource labels for display
resources = [
    "Gate (Reg)",
    "Gate (Digi)",
    "Check-in (Indigo)",
    "Check-in (Akasa)",
    "Security Lanes",
    "Boarding Gates"
]

# Values before & after
before_caps = [
    baseline_T2["gate_reg"],
    baseline_T2["gate_digi"],
    baseline_T2["checkin_indigo"],
    baseline_T2["checkin_akasa"],
    baseline_T2["security"],
    baseline_T2["num_gates"]
]

after_caps = [
    final_T2["gate_reg"],
    final_T2["gate_digi"],
    final_T2["checkin_indigo"],
    final_T2["checkin_akasa"],
    final_T2["security"],
    final_T2["num_gates"]
]

x = np.arange(len(resources))
width = 0.35  # bar width

plt.figure(figsize=(10,5))
bars1 = plt.bar(x - width/2, before_caps, width, label="Before", color="#4C72B0", edgecolor="black")
bars2 = plt.bar(x + width/2, after_caps,  width, label="After",  color="#DD8452", edgecolor="black")

# Add numbers above bars
for b in bars1:
    plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.4, f"{int(b.get_height())}", ha='center', fontsize=10)

for b in bars2:
    plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.4, f"{int(b.get_height())}", ha='center', fontsize=10)

plt.xticks(x, resources, rotation=15, fontsize=11)
plt.ylabel("Number of Counters / Gates", fontsize=12)
plt.title("T2 – Resource Counters Before vs After Optimization", fontsize=14, weight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.35)
plt.legend()
plt.tight_layout()
plt.show()



