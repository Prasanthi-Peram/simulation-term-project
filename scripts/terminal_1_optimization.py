
"""Simulation Run

"""

import simpy
import numpy as np
import pandas as pd
from statistics import mean

# ------------------- PARAMETERS -------------------
RANDOM_SEED = 42
RUN_TIME = 1440  # minutes (24 hours)
P_DIGI = 0.19

NUM_GATES = 23   # boarding gates remain fixed

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

def D_gate_Digi_T1(): return 0.5 + np.random.exponential(0.05)
def D_gate_Reg_T1(t): return get_service_time(service_gate, t)
def D_checkin_T1(t):  return get_service_time(service_checkin, t)
def D_security_T1(t): return get_service_time(service_security, t)
def D_walking():      return np.random.triangular(5.0, 10.0, 10.0)
def D_waiting():      return np.random.uniform(20.0, 40.0)
def D_boarding():     return np.random.triangular(5.0, 8.0, 10.0)

# ------------------- LOAD FLIGHT DATA -------------------
df = pd.read_csv('data/Terminal1 Flight Data.csv') # Uncommented to load the data

# ==========================================================
# ✅ FUNCTION: RUN SIMULATION WITH ANY CAPACITIES
# ==========================================================
def run_simulation(cap_gate_reg, cap_gate_digi, cap_checkin_indigo, cap_checkin_spicejet, cap_security, num_gates, cap_boarding_per_gate):


    global cycle_times, waits, queue_lengths, busy_time, throughput, results_by_airline, system_count

   # ✅ fixed boarding desk capacity per gate (not optimized)

    # Reset metrics
    cycle_times = []
    waits = {"gate_reg": [], "gate_digi": [], "checkin_indigo": [], "checkin_spicejet": [],
             "security": [], "boarding": [], "waiting_hall": []}
    queue_lengths = {"checkin": [], "security": [], "boarding": []}
    busy_time = {"checkin": 0, "security": 0, "boarding": 0}
    throughput = np.zeros(28)
    results_by_airline = {"IndiGo": [], "SpiceJet": []}
    system_count = []

    # Passenger Process
    def passenger(env, pid, airline, gate_reg, gate_digi, checkin_indigo, checkin_spicejet, securities, boardings):
        t0 = env.now
        system_count.append((env.now, +1))

        # GATE SELECTION
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

        # CHECK-IN
        if airline.lower() == "indigo":
            chosen = min(checkin_indigo, key=lambda r: len(r.queue))
        else:
            chosen = min(checkin_spicejet, key=lambda r: len(r.queue))
        queue_lengths["checkin"].append(len(chosen.queue))
        with chosen.request() as req:
            t_req = env.now; yield req
            if airline.lower() == "indigo":
                waits["checkin_indigo"].append(env.now - t_req)
            else:
                waits["checkin_spicejet"].append(env.now - t_req)
            st = D_checkin_T1(env.now); yield env.timeout(st)
            busy_time["checkin"] += st

        # SECURITY
        chosen = min(securities, key=lambda r: len(r.queue))
        queue_lengths["security"].append(len(chosen.queue))
        with chosen.request() as req:
            t_req = env.now; yield req
            waits["security"].append(env.now - t_req)
            st = D_security_T1(env.now); yield env.timeout(st)
            busy_time["security"] += st

        # WAITING AREA
        waiting_time = D_waiting()
        waits["waiting_hall"].append(waiting_time)
        yield env.timeout(D_walking() + waiting_time)

        # BOARDING  ✅ uses num_gates and fixed capacity per gate
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

    # Flight Generator
    def flight_source(env, gate_reg, gate_digi, checkin_indigo, checkin_spicejet, securities, boardings):
        pid = 0
        for _, row in df.iterrows():
            dep_time = row["Dep_time_min"]
            n_passengers = int(row["N_passengers"])
            airline = row["Airline"]
            arr_start = max(0.0, dep_time - 120.0)
            arr_end = max(arr_start, min(RUN_TIME, dep_time - 60.0))

            if n_passengers > 0:
                gap = (arr_end - arr_start) / n_passengers
                arrivals = arr_start + np.cumsum(np.random.exponential(gap, n_passengers))
                arrivals = np.clip(arrivals, arr_start, arr_end)
                arrivals.sort()

                for arr_time in arrivals:
                    yield env.timeout(max(0, arr_time - env.now))
                    pid += 1
                    env.process(passenger(env, pid, airline, gate_reg, gate_digi, checkin_indigo, checkin_spicejet, securities, boardings))

    # Environment + Resources
    env = simpy.Environment()
    gate_reg = [simpy.Resource(env, capacity=1) for _ in range(cap_gate_reg)]
    gate_digi = [simpy.Resource(env, capacity=1) for _ in range(cap_gate_digi)]
    checkin_indigo = [simpy.Resource(env, capacity=1) for _ in range(cap_checkin_indigo)]
    checkin_spicejet = [simpy.Resource(env, capacity=1) for _ in range(cap_checkin_spicejet)]
    securities = [simpy.Resource(env, capacity=1) for _ in range(cap_security)]

    # ✅ number of boarding gates = num_gates
    # ✅ each gate has processing capacity = CAP_BOARDING_PER_GATE (fixed)
    boardings = [simpy.Resource(env, capacity=cap_boarding_per_gate) for _ in range(num_gates)]

    env.process(flight_source(env, gate_reg, gate_digi, checkin_indigo, checkin_spicejet, securities, boardings))
    env.run(until=RUN_TIME + 180)

    def avg(lst): return mean(lst) if lst else 0.0

    return {
    # --- Existing Wait Time Metrics ---
    "total_avg_wait": avg(waits["gate_reg"] + waits["gate_digi"])
                      + avg(waits["checkin_indigo"] + waits["checkin_spicejet"])
                      + avg(waits["security"])
                      + avg(waits["boarding"]),

    "avg_wait_gate": avg(waits["gate_reg"] + waits["gate_digi"]),
    "avg_wait_checkin": avg(waits["checkin_indigo"] + waits["checkin_spicejet"]),
    "avg_wait_security": avg(waits["security"]),
    "avg_wait_boarding": avg(waits["boarding"]),

    # --- ✅ NEW QUEUE LENGTH METRICS ---
    "avg_q_checkin": np.mean(queue_lengths["checkin"]) if queue_lengths["checkin"] else 0.0,
    "avg_q_security": np.mean(queue_lengths["security"]) if queue_lengths["security"] else 0.0,
    "avg_q_boarding": np.mean(queue_lengths["boarding"]) if queue_lengths["boarding"] else 0.0,

    # --- System Performance ---
    "avg_time_in_system": np.mean(cycle_times),
    "passengers_completed": len(cycle_times),

    "utilization": {k: busy_time[k] / RUN_TIME for k in busy_time}
}

print(run_simulation(24, 6, 20, 10, 42, 23,50))

import matplotlib
matplotlib.use("Agg")  # disables live display; you can still save figures

# pip install tqdm

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
from IPython.display import clear_output

# ============================
# TERMINAL-1 CAPACITY OPTIMIZER (Colab Live + Excel + Sweeps)
# ============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output

plt.rcParams["figure.dpi"] = 110

# -------------------- TUNING KNOBS --------------------
N_REP = 3                           # replications per configuration
ABS_IMPROVE_MIN = 0.5               # minutes: minimal absolute improvement to accept
REL_IMPROVE_MIN = 0.02              # fraction (e.g., 0.02 = 2%)

# We optimize these, in this order
RESOURCES_ORDER = [
    "checkin_indigo", "checkin_spicejet", "security",
    "gate_reg", "gate_digi", "num_gates"
]

# Safety caps
MAX_CAPS = {
    "gate_reg": 80,
    "gate_digi": 40,
    "checkin_indigo": 120,
    "checkin_spicejet": 80,
    "security": 160,
    "num_gates": 60              # physical max gates
}

# ---- Fixed per-gate boarding capacity (NOT optimized here; you can set 50–100) ----
CAP_BOARDING_PER_GATE = 50

# Track total wait trend for the dashboard
trend = []


# -------------------- FORMATTING & BOTTLENECK --------------------
def _format_caps(caps, include_board_cap=True):
    base = (f"GReg={caps['gate_reg']}, GDigi={caps['gate_digi']}, "
            f"ChkInd={caps['checkin_indigo']}, ChkSJ={caps['checkin_spicejet']}, "
            f"Sec={caps['security']}, Gates={caps['num_gates']}")
    if include_board_cap:
        base += f", GateCap={CAP_BOARDING_PER_GATE}"
    return base

def _bottleneck_from_waits(avg_waits_dict):
    gate = avg_waits_dict["gate"]
    checkin = avg_waits_dict["checkin"]
    security = avg_waits_dict["security"]
    boarding = avg_waits_dict["boarding"]
    total = max(1e-9, gate + checkin + security + boarding)
    shares = {
        "Gate": gate/total,
        "Check-in": checkin/total,
        "Security": security/total,
        "Boarding": boarding/total
    }
    bottleneck = max(shares.items(), key=lambda kv: kv[1])[0]
    return bottleneck, shares


# -------------------- LIVE DASHBOARD (Colab-friendly) --------------------
def _live_dashboard(stats, iteration):
    clear_output(wait=True)
    waits = stats["avg_waits"]
    bn, shares = _bottleneck_from_waits(waits)

    plt.figure(figsize=(12,4))

    # (1) Stage waits (bar)
    plt.subplot(1,3,1)
    plt.bar(["Gate","Check-in","Security","Boarding"],
            [waits["gate"], waits["checkin"], waits["security"], waits["boarding"]],
            color=['#4e79a7','#f28e2b','#e15759','#59a14f'])
    plt.title("Stage Waiting Times")
    plt.ylabel("Minutes")

    # (2) Waiting composition (pie)
    plt.subplot(1,3,2)
    sizes = [waits["gate"], waits["checkin"], waits["security"], waits["boarding"]]
    plt.pie(sizes, labels=["Gate","Check-in","Security","Boarding"], autopct='%1.1f%%', startangle=90)
    plt.title(f"Bottleneck Mix (Bn: {bn})")

    # (3) Total wait trend
    trend.append(waits["total"])
    plt.subplot(1,3,3)
    plt.plot(trend, '-o', linewidth=2)
    plt.title("Total Avg Wait Across Accepted Steps")
    plt.xlabel("Accepted step #")
    plt.ylabel("Minutes")
    plt.grid(True, alpha=0.3)

    plt.suptitle(f"Optimization Progress – Iteration {iteration}", y=1.03)
    plt.tight_layout()
    plt.show()


# -------------------- AVERAGED RUN WRAPPER --------------------
def _run_avg(capacities, n_rep=N_REP, base_seed=12345):
    """
    Calls your run_simulation(...) multiple times and averages results.
    Your function must be defined elsewhere with signature:
      run_simulation(gate_reg, gate_digi, checkin_indigo, checkin_spicejet, security, num_gates, cap_boarding_per_gate)
    and it must use:
      boardings = [simpy.Resource(env, capacity=cap_boarding_per_gate) for _ in range(num_gates)]
    """
    results = []
    seeds = np.random.SeedSequence(base_seed).spawn(n_rep)
    for i, s in enumerate(seeds):
        np.random.seed(int(s.entropy))
        res = run_simulation(
            capacities["gate_reg"],
            capacities["gate_digi"],
            capacities["checkin_indigo"],
            capacities["checkin_spicejet"],
            capacities["security"],
            capacities["num_gates"],
            CAP_BOARDING_PER_GATE   # <- fixed per-gate boarding capacity
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

    # quick bottleneck print
    bn, shares = _bottleneck_from_waits(avg_waits)
    print(f"   ↳ Bottleneck: {bn.upper()} (shares: gate={shares['Gate']*100:.1f}%, "
          f"checkin={shares['Check-in']*100:.1f}%, security={shares['Security']*100:.1f}%, "
          f"boarding={shares['Boarding']*100:.1f}%)")

    return {
        "avg_waits": avg_waits,
        "avg_cycle": avg_cycle,
        "completed": completed,
        "utilization": utilization,
        "bottleneck": bn,
        "shares": shares
    }


# -------------------- SWEEP + PLOT (for report) --------------------
def sweep_resource(resource_name, start_caps, min_cap=None, max_cap=None, step=1, n_rep=N_REP):
    caps = start_caps.copy()
    if min_cap is None: min_cap = caps[resource_name]
    if max_cap is None: max_cap = MAX_CAPS.get(resource_name, caps[resource_name] + 20)

    rows = []
    for c in tqdm(range(min_cap, max_cap + 1, step), desc=f"Sweep {resource_name}"):
        caps[resource_name] = c
        s = _run_avg(caps, n_rep=n_rep)
        rows.append({
            "resource": resource_name,
            "capacity": c,
            "total_avg_wait": s["avg_waits"]["total"],
            "avg_wait_gate": s["avg_waits"]["gate"],
            "avg_wait_checkin": s["avg_waits"]["checkin"],
            "avg_wait_security": s["avg_waits"]["security"],
            "avg_wait_boarding": s["avg_waits"]["boarding"],
            "avg_cycle": s["avg_cycle"],
            "completed": s["completed"],
            "bottleneck": s["bottleneck"]
        })
    return rows

def plot_sweep(rows, title=None, show_components=True):
    if not rows:
        print("No rows to plot.")
        return
    df = pd.DataFrame(rows)
    x = df["capacity"].values
    plt.figure(figsize=(7,4))
    plt.plot(x, df["total_avg_wait"].values, label="Total avg wait")
    if show_components:
        plt.plot(x, df["avg_wait_gate"].values, label="Gate")
        plt.plot(x, df["avg_wait_checkin"].values, label="Check-in")
        plt.plot(x, df["avg_wait_security"].values, label="Security")
        plt.plot(x, df["avg_wait_boarding"].values, label="Boarding")
    plt.xlabel("Capacity")
    plt.ylabel("Average waiting time (min)")
    plt.title(title or f"Wait vs Capacity — {rows[0]['resource']}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# -------------------- EXCEL SAVE --------------------
def save_history_to_excel(history_rows, filename="Optimization_Results_T1.xlsx"):
    if not history_rows:
        print("No history to save.")
        return
    df_hist = pd.DataFrame(history_rows)
    # choose last 'accept' or final for summary
    summary_candidates = df_hist[df_hist["event"].isin(["accept","final","baseline"])]
    last_row = summary_candidates.iloc[-1].to_dict() if len(summary_candidates) else df_hist.iloc[-1].to_dict()
    df_sum = pd.DataFrame([last_row])
    with pd.ExcelWriter(filename, engine="xlsxwriter") as w:
        df_hist.to_excel(w, index=False, sheet_name="history")
        df_sum.to_excel(w, index=False, sheet_name="summary")
    print(f"📄 Saved optimization history & summary to: {filename}")


# -------------------- GREEDY PLATEAU OPTIMIZER --------------------
def optimize_capacities(
    start_caps=dict(gate_reg=24, gate_digi=6, checkin_indigo=20, checkin_spicejet=10, security=42, num_gates=23),
    abs_improve=ABS_IMPROVE_MIN,
    rel_improve=REL_IMPROVE_MIN,
    resources_order=RESOURCES_ORDER,
    n_rep=N_REP,
    verbose=True,
    save_excel=True,
    excel_filename="Optimization_Results_T1.xlsx"
):
    """
    Tries +1 to each resource in resources_order and accepts only if
    total_avg_wait improves by >= ABS_IMPROVE_MIN minutes OR >= REL_IMPROVE_MIN fraction.
    Stops when a full pass yields no accepted increments (plateau).
    """
    def _cap_ok(res, val): return val <= MAX_CAPS.get(res, val)
    caps = start_caps.copy()

    history_rows = []
    accepted_total_waits = []  # trend of accepted states
    iteration = 0

    # Baseline
    base_stats = _run_avg(caps, n_rep=n_rep)
    best_stats = base_stats
    best_obj = best_stats["avg_waits"]["total"]
    bn, _shares = _bottleneck_from_waits(best_stats["avg_waits"])

    if verbose:
        print(f"[Baseline] {_format_caps(caps)} -> TotalAvgWait={best_obj:.3f} min")
    history_rows.append({
        "event": "baseline",
        **caps,
        "cap_per_gate": CAP_BOARDING_PER_GATE,
        "total_avg_wait": best_obj,
        "gate": base_stats["avg_waits"]["gate"],
        "checkin": base_stats["avg_waits"]["checkin"],
        "security": base_stats["avg_waits"]["security"],
        "boarding": base_stats["avg_waits"]["boarding"],
        "avg_cycle": base_stats["avg_cycle"],
        "completed": base_stats["completed"],
        "bottleneck": bn
    })
    accepted_total_waits.append(best_obj)
    _live_dashboard(base_stats, iteration=iteration)

    improved_any = True
    while improved_any:
        improved_any = False
        for res in tqdm(resources_order, desc="Resources pass"):
            local_improved = False
            while True:
                new_cap = caps[res] + 1
                if not _cap_ok(res, new_cap):
                    if verbose:
                        print(f"[Limit] {res} cannot exceed {MAX_CAPS[res]}")
                    break

                trial_caps = caps.copy()
                trial_caps[res] = new_cap
                if verbose:
                    print(f"\n→ Trying {res} = {new_cap}  ({_format_caps(trial_caps)})")

                trial_stats = _run_avg(trial_caps, n_rep=n_rep)
                trial_obj = trial_stats["avg_waits"]["total"]

                abs_gain = best_obj - trial_obj
                rel_gain = abs_gain / best_obj if best_obj > 0 else 0.0

                bn_t, shares_t = _bottleneck_from_waits(trial_stats["avg_waits"])
                history_rows.append({
                    "event": "trial",
                    **trial_caps,
                    "cap_per_gate": CAP_BOARDING_PER_GATE,
                    "total_avg_wait": trial_obj,
                    "gate": trial_stats["avg_waits"]["gate"],
                    "checkin": trial_stats["avg_waits"]["checkin"],
                    "security": trial_stats["avg_waits"]["security"],
                    "boarding": trial_stats["avg_waits"]["boarding"],
                    "avg_cycle": trial_stats["avg_cycle"],
                    "completed": trial_stats["completed"],
                    "bottleneck": bn_t,
                    "abs_gain": abs_gain,
                    "rel_gain": rel_gain
                })

                # Show trial state for context
                _live_dashboard(trial_stats, iteration=max(1, iteration)+0.1)  # fractional to indicate trial

                # Accept?
                if (abs_gain >= abs_improve) or (rel_gain >= rel_improve):
                    caps = trial_caps
                    best_stats = trial_stats
                    best_obj = trial_obj
                    iteration += 1
                    accepted_total_waits.append(best_obj)
                    local_improved = True
                    improved_any = True
                    if verbose:
                        print(f"[Improve] {res:>16} -> {caps[res]:>3} | TotalAvgWait={best_obj:.3f} "
                              f"(abs_gain={abs_gain:.3f}, rel={rel_gain*100:.2f}%)")

                    history_rows.append({
                        "event": "accept",
                        **caps,
                        "cap_per_gate": CAP_BOARDING_PER_GATE,
                        "total_avg_wait": best_obj,
                        "gate": best_stats["avg_waits"]["gate"],
                        "checkin": best_stats["avg_waits"]["checkin"],
                        "security": best_stats["avg_waits"]["security"],
                        "boarding": best_stats["avg_waits"]["boarding"],
                        "avg_cycle": best_stats["avg_cycle"],
                        "completed": best_stats["completed"],
                        "bottleneck": bn_t
                    })

                    # Update dashboard with accepted state
                    _live_dashboard(best_stats, iteration=iteration)

                    # continue with same resource
                else:
                    if verbose:
                        print(f"[Plateau] {res:>16} at {caps[res]} (no meaningful gain)")
                    break  # plateau for this resource

            if verbose and not local_improved:
                print(f"[Plateau] {res:>16} at {caps[res]} (no meaningful gain)")

    # Final report prints
    if verbose:
        print("\n--- OPTIMIZED CAPACITIES ---")
        print(_format_caps(caps))
        print(f"TotalAvgWait={best_obj:.3f} min")
        print("Avg waits (min):",
              f"Gate={best_stats['avg_waits']['gate']:.2f},",
              f"Check-in={best_stats['avg_waits']['checkin']:.2f},",
              f"Security={best_stats['avg_waits']['security']:.2f},",
              f"Boarding={best_stats['avg_waits']['boarding']:.2f}")
        print("Utilization:",
              ", ".join([f"{k}={best_stats['utilization'][k]:.3f}" for k in best_stats["utilization"]]))
        print(f"Avg time in system = {best_stats['avg_cycle']:.2f} min; "
              f"Completed pax ≈ {best_stats['completed']}")

    # Final log for Excel
    history_rows.append({
        "event": "final",
        **caps,
        "cap_per_gate": CAP_BOARDING_PER_GATE,
        "total_avg_wait": best_obj,
        "gate": best_stats["avg_waits"]["gate"],
        "checkin": best_stats["avg_waits"]["checkin"],
        "security": best_stats["avg_waits"]["security"],
        "boarding": best_stats["avg_waits"]["boarding"],
        "avg_cycle": best_stats["avg_cycle"],
        "completed": best_stats["completed"],
        "bottleneck": best_stats["bottleneck"]
    })

    if save_excel:
        save_history_to_excel(history_rows, filename=excel_filename)

    return caps, best_stats, history_rows


# ======================
# RUN OPTIMIZER (example)
# ======================
if __name__ == "__main__":
    start_caps = dict(
        gate_reg=24, gate_digi=6,
        checkin_indigo=20, checkin_spicejet=10,
        security=42, num_gates=23
    )

    best_caps, best_stats, history_rows = optimize_capacities(
        start_caps=start_caps,
        abs_improve=ABS_IMPROVE_MIN,
        rel_improve=REL_IMPROVE_MIN,
        resources_order=RESOURCES_ORDER,
        n_rep=N_REP,
        verbose=True,
        save_excel=True,
        excel_filename="Optimization_Results_T1.xlsx"
    )

    # Optional sweeps for report-quality curves:
    # sec_rows = sweep_resource("security", best_caps,
    #                           min_cap=max(10, best_caps["security"]-5),
    #                           max_cap=best_caps["security"]+8,
    #                           step=1, n_rep=max(1, N_REP-1))
    # plot_sweep(sec_rows, title="Security – Wait vs Capacity", show_components=True)

    # gates_rows = sweep_resource("num_gates", best_caps,
    #                             min_cap=max(10, best_caps["num_gates"]-5),
    #                             max_cap=best_caps["num_gates"]+10,
    #                             step=1, n_rep=max(1, N_REP-1))
    # plot_sweep(gates_rows, title="Number of Gates – Wait vs Capacity", show_components=True)

# pip install xlsxwriter

import matplotlib.pyplot as plt
import numpy as np

# Extract baseline & final rows
df = pd.DataFrame(history_rows)
baseline = df[df.event=="baseline"].iloc[-1]
final = df[df.event=="final"].iloc[-1]

labels = ["Gate", "Check-in", "Security", "Boarding"]
before = [baseline["gate"], baseline["checkin"], baseline["security"], baseline["boarding"]]
after  = [final["gate"], final["checkin"], final["security"], final["boarding"]]

x = np.arange(len(labels))
width = 0.38  # width of bars

plt.figure(figsize=(8,5))
plt.bar(x - width/2, before, width, color="#4C72B0", edgecolor="black", label="Before")
plt.bar(x + width/2, after, width, color="#DD8452", edgecolor="black", label="After")

plt.xticks(x, labels, fontsize=12)
plt.ylabel("Average Waiting Time (min)", fontsize=12)
plt.title("Before vs After Optimization – Stage-wise Waiting Times", fontsize=14, weight="bold")

plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.35)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

labels = ["Gate", "Check-in", "Security", "Boarding"]

# Convert waits into shares (so pie slices sum to 100%)
before_share = np.array(before) / np.sum(before)
after_share  = np.array(after)  / np.sum(after)

# Identify bottleneck (largest share) before/after
explode_before = [0.08 if v == max(before_share) else 0 for v in before_share]
explode_after  = [0.08 if v == max(after_share)  else 0 for v in after_share]

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]  # pleasant & high-contrast

fig, ax = plt.subplots(1, 2, figsize=(10,5))

ax[0].pie(before_share, labels=labels, autopct="%1.1f%%", explode=explode_before,
          startangle=90, colors=colors, shadow=True)
ax[0].set_title("Before Optimization", fontsize=13)

ax[1].pie(after_share, labels=labels, autopct="%1.1f%%", explode=explode_after,
          startangle=90, colors=colors, shadow=True)
ax[1].set_title("After Optimization", fontsize=13)

plt.suptitle("Bottleneck Stage Share Comparison (Before vs After)", fontsize=15, weight='bold')
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

resources = ["Gate (Reg)", "Gate (Digi)", "Check-in (IndiGo)", "Check-in (SpiceJet)", "Security Lanes", "Boarding Gates"]

before_caps = [
    baseline["gate_reg"], baseline["gate_digi"],
    baseline["checkin_indigo"], baseline["checkin_spicejet"],
    baseline["security"], baseline["num_gates"]
]

after_caps = [
    final["gate_reg"], final["gate_digi"],
    final["checkin_indigo"], final["checkin_spicejet"],
    final["security"], final["num_gates"]
]

x = np.arange(len(resources))
width = 0.35  # width of the bars

fig, ax = plt.subplots(figsize=(9,5))
bars1 = ax.bar(x - width/2, before_caps, width, label="Before", color="#4C72B0")
bars2 = ax.bar(x + width/2, after_caps, width, label="After", color="#DD8452")

# Add value labels on top
for b in bars1:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f"{int(b.get_height())}", ha='center', fontsize=10)

for b in bars2:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f"{int(b.get_height())}", ha='center', fontsize=10)

ax.set_ylabel("Number of Active Counters / Gates", fontsize=12)
ax.set_title("Comparison of Resource Capacities Before and After Optimization", fontsize=14, weight='bold')
ax.set_xticks(x)
ax.set_xticklabels(resources, rotation=20)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.35)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

resources_u = ["Check-in", "Security"]

util_before = [
    baseline["checkin"],
    baseline["security"],
]

util_after = [
    final["checkin"],
    final["security"],
]

x = np.arange(len(resources_u))
width = 0.35  # bar width

fig, ax = plt.subplots(figsize=(9,5))

bars1 = ax.bar(x - width/2, util_before, width, label="Before", color="#4C72B0")
bars2 = ax.bar(x + width/2, util_after,  width, label="After",  color="#DD8452")

# Add value labels above bars
for b in bars1 + bars2:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
            f"{b.get_height():.2f}", ha='center', fontsize=10)

ax.set_ylabel("Utilization (Busy Fraction)", fontsize=12)
ax.set_title("Resource Utilization Before vs After Optimization", fontsize=14, weight='bold')
ax.set_xticks(x)
ax.set_xticklabels(resources_u, fontsize=11)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.35)

plt.tight_layout()
plt.show()

df_acc = df[df.event=="accept"]
plt.figure(figsize=(7,4))
plt.plot(df_acc.total_avg_wait.values, '-o')
plt.xlabel("Accepted Improvement Step")
plt.ylabel("Total Avg Wait (min)")
plt.title("Convergence Trend During Optimization")
plt.grid(alpha=0.4)
plt.show()

# ============================
# TERMINAL-1 CAPACITY OPTIMIZER (Colab Version with Live Dashboard)
# ============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output

plt.rcParams["figure.dpi"] = 110

# -----------------------------------
# TUNING PARAMETERS
# -----------------------------------
N_REP = 3
ABS_IMPROVE_MIN = 0.5
REL_IMPROVE_MIN = 0.02

RESOURCES_ORDER = [
    "checkin_indigo", "checkin_spicejet", "security",
    "gate_reg", "gate_digi", "num_gates"
]

MAX_CAPS = {
    "gate_reg": 80,
    "gate_digi": 40,
    "checkin_indigo": 120,
    "checkin_spicejet": 80,
    "security": 160,
    "num_gates": 60
}

# TRACK GLOBAL TREND FOR DASHBOARD PLOT 3
trend = []


# -------------------------------------------------
# Bottleneck Detection
# -------------------------------------------------
def _bottleneck(avg_waits):
    waits = {
        "Gate": avg_waits["gate"],
        "Check-in": avg_waits["checkin"],
        "Security": avg_waits["security"],
        "Boarding": avg_waits["boarding"],
    }
    return max(waits, key=waits.get), waits


# -------------------------------------------------
# Live Dashboard (works in Google Colab)
# -------------------------------------------------
def _live_dashboard(stats, iteration):
    clear_output(wait=True)
    waits = stats["avg_waits"]

    plt.figure(figsize=(12,4))

    # (1) Bar chart for stage waiting times
    plt.subplot(1,3,1)
    plt.bar(["Gate","Check-in","Security","Boarding"],
            [waits["gate"], waits["checkin"], waits["security"], waits["boarding"]],
            color=['#4e79a7','#f28e2b','#e15759','#59a14f'])
    plt.title("Stage Waiting Times")
    plt.ylabel("Minutes")

    # (2) Pie chart for bottleneck contribution
    plt.subplot(1,3,2)
    sizes = [waits["gate"], waits["checkin"], waits["security"], waits["boarding"]]
    plt.pie(sizes, labels=["Gate","Check-in","Security","Boarding"], autopct='%1.1f%%')
    plt.title("Waiting Composition (Bottleneck Mix)")

    # (3) Trend line of total avg wait across iterations
    trend.append(waits["total"])
    plt.subplot(1,3,3)
    plt.plot(trend, '-o', linewidth=2)
    plt.title("Total Avg Wait Across Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Minutes")

    plt.suptitle(f"Optimization Progress – Iteration {iteration}")
    plt.tight_layout()
    plt.show()


# -------------------------------------------------
# Averaged Simulation Wrapper (unchanged logic)
# -------------------------------------------------
def _run_avg(caps, n_rep=N_REP, base_seed=12345):
    results = []
    seeds = np.random.SeedSequence(base_seed).spawn(n_rep)

    for i, s in enumerate(seeds):
        np.random.seed(int(s.entropy))
        r = run_simulation(
            caps["gate_reg"],
            caps["gate_digi"],
            caps["checkin_indigo"],
            caps["checkin_spicejet"],
            caps["security"],
            caps["num_gates"]
        )
        results.append(r)

    def A(key): return float(np.mean([x[key] for x in results]))

    avg_waits = {
        "gate": A("avg_wait_gate"),
        "checkin": A("avg_wait_checkin"),
        "security": A("avg_wait_security"),
        "boarding": A("avg_wait_boarding"),
        "total": A("total_avg_wait"),
    }

    return {
        "avg_waits": avg_waits,
        "avg_cycle": A("avg_time_in_system"),
        "completed": int(np.mean([x["passengers_completed"] for x in results])),
        "utilization": {k: float(np.mean([x["utilization"][k] for x in results])) for k in results[0]["utilization"]}
    }


# -------------------------------------------------
# MAIN OPTIMIZER
# -------------------------------------------------
def optimize_capacities(start_caps, save_excel=True, excel_filename="Optimization_Results_T1.xlsx"):

    caps = start_caps.copy()
    history = []

    stats = _run_avg(caps)
    best_wait = stats["avg_waits"]["total"]
    history.append({"event": "baseline", **caps, "total_avg_wait": best_wait})
    print(f"\n[Baseline] {caps} → TotalAvgWait = {best_wait:.2f} min")
    _live_dashboard(stats, iteration=0)

    improved = True
    iteration = 1

    while improved:
        improved = False

        for res in tqdm(RESOURCES_ORDER, desc="Resource sweep"):
            while True:
                new_cap = caps[res] + 1
                if new_cap > MAX_CAPS[res]:
                    break

                test_caps = caps.copy()
                test_caps[res] = new_cap
                test_stats = _run_avg(test_caps)
                test_wait = test_stats["avg_waits"]["total"]

                abs_gain = best_wait - test_wait
                rel_gain = abs_gain / best_wait if best_wait > 0 else 0

                history.append({"event": "trial", **test_caps, "total_avg_wait": test_wait})

                if abs_gain >= ABS_IMPROVE_MIN or rel_gain >= REL_IMPROVE_MIN:
                    caps = test_caps
                    stats = test_stats
                    best_wait = test_wait
                    improved = True

                    print(f"[Improve] {res} → {caps[res]}  | New TotalAvgWait = {best_wait:.2f} min")
                    _live_dashboard(stats, iteration)
                    iteration += 1
                else:
                    break

    print("\n✅ OPTIMIZATION COMPLETE")
    print("Optimal Capacities:", caps)
    print(f"Final TotalAvgWait = {best_wait:.2f} min")

    if save_excel:
        pd.DataFrame(history).to_excel(excel_filename, index=False)
        print(f"📄 Saved full optimization history → {excel_filename}")

    return caps, stats, history


# -------------------------------------------------
# RUN OPTIMIZER
# -------------------------------------------------
start_caps = dict(
    gate_reg=24, gate_digi=6,
    checkin_indigo=20, checkin_spicejet=10,
    security=42, num_gates=23
)

best_caps, best_stats, history = optimize_capacities(start_caps)

"""Arrival Rate graph

"""

# Calculate passenger arrival rate per hour
df['Arrival_time_min_start'] = df['Dep_time_min'] - 120
df['Arrival_time_min_end'] = df['Dep_time_min'] - 60

# Ensure arrival times are within RUN_TIME
df['Arrival_time_min_start'] = df['Arrival_time_min_start'].clip(lower=0)
df['Arrival_time_min_end'] = df['Arrival_time_min_end'].clip(upper=RUN_TIME)

# Calculate passengers arriving per minute for each flight
df['Passengers_per_min'] = df['N_passengers'] / (df['Arrival_time_min_end'] - df['Arrival_time_min_start'] + 1e-9) # Add small epsilon to avoid division by zero

# Create an array to store arrival rate per minute over the RUN_TIME
arrival_rate_per_min = np.zeros(RUN_TIME + 1)

# Distribute passengers over their arrival window
for index, row in df.iterrows():
    start = int(row['Arrival_time_min_start'])
    end = int(row['Arrival_time_min_end'])
    passengers = row['Passengers_per_min']
    if start <= end:
        arrival_rate_per_min[start:end+1] += passengers

# Aggregate arrival rate per hour
arrival_rate_per_hour = [arrival_rate_per_min[i:i+60].sum() for i in range(0, RUN_TIME, 60)]

# Use the correct variable name for plotting
arrival_rate = arrival_rate_per_hour

# === PLOT: Arrival Rate Per Hour ===
import matplotlib.pyplot as plt

plt.figure(figsize=(9,4))
plt.plot(range(len(arrival_rate_per_hour)), arrival_rate_per_hour, linewidth=2)
plt.xlabel("Hour of day")
plt.ylabel("Number of passengers")
plt.title("Passenger Arrival Rate Per Hour – Terminal 1")
plt.grid(True)
plt.show()

"""Opt with Queues

"""

# ============================
# TERMINAL-1 QUEUE OPTIMIZER
# ============================

import numpy as np
import pandas as pd
from tqdm import tqdm

N_REP = 3
ABS_IMPROVE_MIN = 0.25     # smaller threshold works better for queue length optimization
REL_IMPROVE_MIN = 0.01
SHOW_PROGRESS = True

RESOURCES_ORDER = [
    "checkin_indigo", "checkin_spicejet", "security",
    "gate_reg", "gate_digi", "num_gates"
]

MAX_CAPS = {
    "gate_reg": 80,
    "gate_digi": 40,
    "checkin_indigo": 120,
    "checkin_spicejet": 80,
    "security": 160,
    "num_gates": 60
}

# Add CAP_BOARDING_PER_GATE as a global variable, or get it from the other cell
# For now, assuming it's defined elsewhere globally or in a previous cell
# If not, we might need to add a cell to define it.
# Let's assume it's available as CAP_BOARDING_PER_GATE from cell u7YJGxnKHVwt

def _run_avg(caps, n_rep=N_REP, base_seed=12345):
    results = []
    seeds = np.random.SeedSequence(base_seed).spawn(n_rep)

    iterator = range(n_rep)
    if SHOW_PROGRESS:
        iterator = tqdm(iterator, leave=False, desc=f"Sim (caps={caps})")

    for i in iterator:
        np.random.seed(int(seeds[i].entropy))
        r = run_simulation(
            caps["gate_reg"],
            caps["gate_digi"],
            caps["checkin_indigo"],
            caps["checkin_spicejet"],
            caps["security"],
            caps["num_gates"],
            CAP_BOARDING_PER_GATE  # Pass the missing argument
        )
        results.append(r)

    def _avg(key):
        return float(np.mean([res[key] for res in results]))

    avg_queues = {
        "checkin": _avg("avg_q_checkin"),
        "security": _avg("avg_q_security"),
        "boarding": _avg("avg_q_boarding"),
        "total": _avg("avg_q_checkin") + _avg("avg_q_security") + _avg("avg_q_boarding")
    }

    avg_waits = {
        "total": _avg("total_avg_wait"),
        "gate": _avg("avg_wait_gate"),
        "checkin": _avg("avg_wait_checkin"),
        "security": _avg("avg_wait_security"),
        "boarding": _avg("avg_wait_boarding"),
    }

    return {
        "avg_queues": avg_queues,
        "avg_waits": avg_waits,
        "avg_time_in_system": _avg("avg_time_in_system"),
        "completed": int(np.mean([res["passengers_completed"] for res in results])),
    }


def optimize_capacities(start_caps):
    caps = start_caps.copy()
    best = _run_avg(caps)
    best_obj = best["avg_queues"]["total"]

    print(f"[Baseline] {caps} -> TotalAvgQueue={best_obj:.3f}")

    improved_any = True
    history = [("baseline", caps.copy(), best_obj)]

    while improved_any:
        improved_any = False

        for res in RESOURCES_ORDER:
            while True:
                new_val = caps[res] + 1
                if new_val > MAX_CAPS[res]:
                    break

                trial = caps.copy()
                trial[res] = new_val

                stats = _run_avg(trial)
                obj = stats["avg_queues"]["total"]

                abs_gain = best_obj - obj
                rel_gain = abs_gain / best_obj if best_obj > 0 else 0

                if abs_gain >= ABS_IMPROVE_MIN or rel_gain >= REL_IMPROVE_MIN:
                    caps = trial
                    best = stats
                    best_obj = obj
                    improved_any = True
                    history.append((res, new_val, best_obj))
                    print(f"[Improve] {res} -> {new_val} | Queue={best_obj:.3f} (gain={abs_gain:.3f})")
                else:
                    print(f"[Plateau] {res} at {caps[res]}")
                    break

    print("\n---- FINAL OPTIMIZED CONFIG ----")
    print(caps)
    print(f"Total Avg Queue Length = {best_obj:.3f}")
    print(best)
    return caps, best, history


# RUN
start_caps = dict(
    gate_reg=24, gate_digi=6,
    checkin_indigo=20, checkin_spicejet=10,
    security=42, num_gates=23
)

best_caps, best_stats, history = optimize_capacities(start_caps)

