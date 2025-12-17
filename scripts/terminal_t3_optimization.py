# =========================
# SIMULATION – TERMINAL 3
# =========================
# Ready-to-run script:
# - Uses one combined Check-in pool (Option A)
# - Loads CSV: 'data/Terminal3 Flight Data.csv'
# - Prints detailed report + saves SIMAN-style text report

import simpy
import numpy as np
import pandas as pd
import time
from statistics import mean

# ------------------- PARAMETERS -------------------
RANDOM_SEED = 42
RUN_TIME = 1440           # minutes (24 hours)
P_DIGI = 0.19             # % passengers using Digi
P_DOMESTIC = 0.63         # share of domestic at security

# Capacities (you can tweak)
CAP_GATE_REG = 16
CAP_GATE_DIGI = 6
CAP_CHECKIN = 75          # single combined check-in pool (Option A)
CAP_SECURITY = 64         # total security counters
CAP_BOARDING = 50         # capacity per boarding gate (servers per gate)
NUM_GATES = 62            # total boarding gates

np.random.seed(42)
start_time = time.time()

# ------------------- TIME INTERVALS -------------------
time_blocks = np.array([
    0, 60, 120, 180, 240, 300, 315, 330, 345, 360, 375, 390, 405,
    420, 435, 450, 465, 480, 495, 510, 525, 540, 600, 660, 720, 780,
    840, 900, 960, 975, 990, 1005, 1020, 1035, 1050, 1065, 1080,
    1095, 1110, 1125, 1140, 1155, 1170, 1185, 1200, 1260, 1320, 1380
])

# ------------------- SERVICE TIME ARRAYS (means, minutes) -------------------
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

# ------------------- TIME-DEPENDENT SAMPLERS -------------------
def get_service_time(arr, t):
    idx = np.searchsorted(time_blocks, t, side="right") - 1
    idx = max(0, min(idx, len(arr) - 1))
    return np.random.exponential(arr[idx])

def D_gate_Digi(): return 0.5 + np.random.exponential(0.05)
def D_gate_Reg(t): return get_service_time(service_gate, t)
def D_checkin(t):  return get_service_time(service_checkin, t)
def D_security_domestic(t): return get_service_time(service_domestic_security, t)
def D_security_int_first(t): return get_service_time(service_international_security, t)
def D_security_int_econ(t):  return get_service_time(service_international_security, t)
def D_walking():      return np.random.triangular(5.0, 10.0, 10.0)
def D_waiting():      return np.random.uniform(20.0, 40.0)
def D_boarding():     return np.random.triangular(5.0, 8.0, 10.0)


def run_simulation_T3(cap_gate_reg, cap_gate_digi, cap_checkin, cap_security,
                      num_gates, cap_boarding_per_gate, df):

    import simpy
    import numpy as np
    from statistics import mean

    P_DIGI = 0.19
    P_DOMESTIC = 0.63
    RUN_TIME = 1440

    # Boarding gate allocation
    num_gates_dom = max(1, int(round(num_gates * P_DOMESTIC)))
    num_gates_int = max(1, num_gates - num_gates_dom)

    # ------------------ Metrics ------------------
    cycle_times = []
    waits = {"gate_reg": [], "gate_digi": [], "checkin": [],
             "security_dom": [], "security_int_first": [], "security_int_econ": [],
             "boarding": []}

    def avg(a): return mean(a) if a else 0

    # ------------------ Passenger Process ------------------
    def passenger(env, ptype, gate_reg, gate_digi, checkins, sec_dom, sec_int, board_dom, board_int):
        t0 = env.now

        # Gate
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

        # Check-in
        chosen = min(checkins, key=lambda r: len(r.queue))
        with chosen.request() as req:
            t_req = env.now; yield req
            waits["checkin"].append(env.now - t_req)
            yield env.timeout(D_checkin(env.now))

        # Security + Boarding
        if ptype == "D":
            chosen = min(sec_dom, key=lambda r: len(r.queue))
            with chosen.request() as req:
                t_req = env.now; yield req
                waits["security_dom"].append(env.now - t_req)
                yield env.timeout(D_security_domestic(env.now))

            chosen = min(board_dom, key=lambda r: len(r.queue))
            with chosen.request() as req:
                t_req = env.now; yield req
                waits["boarding"].append(env.now - t_req)
                yield env.timeout(D_boarding())

        else:
            chosen = min(sec_int, key=lambda r: len(r.queue))
            with chosen.request() as req:
                t_req = env.now; yield req
                if np.random.rand() < 0.20:
                    waits["security_int_first"].append(env.now - t_req)
                    yield env.timeout(D_security_int_first(env.now))
                else:
                    waits["security_int_econ"].append(env.now - t_req)
                    yield env.timeout(D_security_int_econ(env.now))

            chosen = min(board_int, key=lambda r: len(r.queue))
            with chosen.request() as req:
                t_req = env.now; yield req
                waits["boarding"].append(env.now - t_req)
                yield env.timeout(D_boarding())

        # Waiting area + walk
        yield env.timeout(D_walking() + D_waiting())

        cycle_times.append(env.now - t0)

    # ------------------ Flight Generator ------------------
    def flight_source(env, gate_reg, gate_digi, checkins, sec_dom, sec_int, board_dom, board_int):
        for _, row in df.iterrows():
            dep = row["Dep_time_min"]
            n = int(row["N_passengers"])
            ptype = row["Type"]
            if n <= 0: continue

            arr_start, arr_end = max(0, dep-120), max(0, dep-60)
            if arr_end <= arr_start: continue

            mean_gap = (arr_end-arr_start)/n
            arrivals = arr_start + np.cumsum(np.random.exponential(mean_gap, n))
            arrivals = np.clip(arrivals, arr_start, arr_end)

            for t in arrivals:
                yield env.timeout(max(0, t - env.now))
                env.process(passenger(env, ptype, gate_reg, gate_digi, checkins, sec_dom, sec_int, board_dom, board_int))

    # ------------------ Resources ------------------
    env = simpy.Environment()

    gate_reg = [simpy.Resource(env, capacity=1) for _ in range(cap_gate_reg)]
    gate_digi = [simpy.Resource(env, capacity=1) for _ in range(cap_gate_digi)]
    checkins = [simpy.Resource(env, capacity=1) for _ in range(cap_checkin)]

    sec_dom = [simpy.Resource(env, capacity=1) for _ in range(int(cap_security*0.63))]
    sec_int = [simpy.Resource(env, capacity=1) for _ in range(cap_security - int(cap_security*0.63))]

    board_dom = [simpy.Resource(env, capacity=cap_boarding_per_gate) for _ in range(num_gates_dom)]
    board_int = [simpy.Resource(env, capacity=cap_boarding_per_gate) for _ in range(num_gates_int)]

    env.process(flight_source(env, gate_reg, gate_digi, checkins, sec_dom, sec_int, board_dom, board_int))
    env.run(until=RUN_TIME + 180)

    # ------------------ Return Summary ------------------
    return {
        "avg_wait_gate": avg(waits["gate_reg"] + waits["gate_digi"]),
        "avg_wait_checkin": avg(waits["checkin"]),
        "avg_wait_security": avg(waits["security_dom"] + waits["security_int_first"] + waits["security_int_econ"]),
        "avg_wait_boarding": avg(waits["boarding"]),
        "avg_time_in_system": np.mean(cycle_times),
        "passengers_completed": len(cycle_times)
    }

df3 = pd.read_csv('data/Terminal3 Flight Data.csv')

result_T3 = run_simulation_T3(
    cap_gate_reg=16,
    cap_gate_digi=6,
    cap_checkin=75,
    cap_security=64,
    num_gates=62,
    cap_boarding_per_gate=50,
    df=df3
)

print(result_T3)

import matplotlib
matplotlib.use("Agg")  # disables live display; you can still save figures

# pip install tqdm

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
from IPython.display import clear_output

import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 120

# ============================
# TERMINAL-3 OPTIMIZER + BOTTLENECK HISTORY PLOT (Increase-only)
# ============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm

plt.rcParams["figure.dpi"] = 110

# -------------------- RANDOMNESS CONTROL --------------------
np.random.seed(42)

# -------------------- TUNING KNOBS --------------------
N_REP = 3
ABS_IMPROVE_MIN = 0.5
REL_IMPROVE_MIN = 0.02

RESOURCES_ORDER_T3 = ["checkin", "security", "gate_reg", "gate_digi", "num_gates"]

MAX_CAPS_T3 = {
    "gate_reg": 60,
    "gate_digi": 30,
    "checkin": 120,
    "security": 120,
    "num_gates": 62
}
CAP_BOARDING_PER_GATE = 50

# runtime state
MIN_CAPS_T3 = {}
trend = []                 # total wait values at each accepted step
bn_history = []            # bottleneck label at each accepted step (incl. baseline)
stage_waits_history = []   # (gate, checkin, security, boarding) at each accepted step (incl. baseline)


# -------------------- HELPERS --------------------
def _format_caps(c):
    return f"GReg={c['gate_reg']}, GDigi={c['gate_digi']}, Chk={c['checkin']}, Sec={c['security']}, Gates={c['num_gates']}"

def _bottleneck_label(waits_dict):
    # waits_dict has keys: gate, checkin, security, boarding
    shares = {
        "Gate": waits_dict["gate"],
        "Check-in": waits_dict["checkin"],
        "Security": waits_dict["security"],
        "Boarding": waits_dict["boarding"],
    }
    return max(shares, key=shares.get)


# -------------------- LIVE DASHBOARD (3-panel) --------------------
def _live(stats, step):
    clear_output(wait=True)

    waits = stats["avg_waits"]
    gate = waits["gate"]; checkin = waits["checkin"]; security = waits["security"]; boarding = waits["boarding"]
    total = waits["total"]
    trend.append(total)

    bn = _bottleneck_label(waits)

    plt.figure(figsize=(14,4))

    # (1) Stage waits
    plt.subplot(1,3,1)
    plt.bar(["Gate","Check-in","Security","Boarding"], [gate,checkin,security,boarding], lw=0)
    plt.title("Stage Waiting Times")
    plt.ylabel("Minutes")

    # (2) Bottleneck mix
    plt.subplot(1,3,2)
    plt.pie([gate, checkin, security, boarding], labels=["Gate","Check-in","Security","Boarding"],
            autopct='%1.1f%%', startangle=90)
    plt.title(f"Bottleneck: {bn}")

    # (3) Accepted trend
    plt.subplot(1,3,3)
    plt.plot(trend, '-o', linewidth=2)
    plt.title("Total Avg Wait Across Accepted Steps")
    plt.xlabel("Accepted step #")
    plt.ylabel("Minutes")
    plt.grid(True, alpha=0.3)

    plt.suptitle(f"T3 Optimization Progress – Iteration {step}", y=1.05)
    plt.tight_layout()
    plt.show()


# -------------------- RUN MULTIPLE REPS --------------------
def _run_avg_t3(cap):
    runs=[]
    for _ in range(N_REP):
        runs.append(run_simulation_T3(
            cap["gate_reg"], cap["gate_digi"], cap["checkin"], cap["security"],
            cap["num_gates"], CAP_BOARDING_PER_GATE, df3
        ))
    def A(k): return float(np.mean([r[k] for r in runs]))

    gate=A("avg_wait_gate"); checkin=A("avg_wait_checkin"); security=A("avg_wait_security"); boarding=A("avg_wait_boarding")
    total = gate + checkin + security + boarding
    return dict(avg_waits=dict(gate=gate,checkin=checkin,security=security,boarding=boarding,total=total))


# -------------------- OPTIMIZER (increase-only, +1/+2/+3) --------------------
def optimize_T3(start_caps):
    global MIN_CAPS_T3, trend
    trend=[]

    history_caps = []
    history_stats = []

    MIN_CAPS_T3 = start_caps.copy()

    caps=start_caps.copy()
    best=_run_avg_t3(caps)
    best_obj=best["avg_waits"]["total"]
    step=0

    print(f"\nBASELINE → Total Waiting Time = {best_obj:.2f} min")
    _live(best,step)

    # store baseline
    history_caps.append(caps.copy())
    history_stats.append(best)

    for res in RESOURCES_ORDER_T3:
        print(f"\n=== Trying to Increase {res.upper()} ===")
        while True:
            improved=False
            for inc in [1,2,3]:
                trial=caps.copy()
                trial[res]+=inc
                if trial[res] < MIN_CAPS_T3[res]: continue
                if trial[res] > MAX_CAPS_T3[res]: continue

                trial_stats=_run_avg_t3(trial)
                new=trial_stats["avg_waits"]["total"]
                gain = best_obj - new

                print(f"{res} = {trial[res]} (+{inc}) → Total Wait {new:.2f} (gain {gain:.2f})")

                if (gain >= ABS_IMPROVE_MIN) or (gain/max(best_obj,1e-9) >= REL_IMPROVE_MIN):
                    caps=trial; best_obj=new; best=trial_stats; step+=1; improved=True
                    print(f" ✅ Accepted → {res} now {caps[res]}")
                    _live(best,step)

                    history_caps.append(caps.copy())
                    history_stats.append(best)
                    break

            if not improved:
                print(f" ❌ No further improvement → stopping {res}")
                break

    print("\n✅ FINAL OPTIMAL CAPACITY SETTINGS:")
    print(_format_caps(caps))
    print(f"Final Total Waiting Time = {best_obj:.2f} min")

    return caps, best, history_caps, history_stats

# -------------------- RUN --------------------
start_caps_T3 = dict(gate_reg=16, gate_digi=6, checkin=75, security=64, num_gates=62)
best_caps, best_stats, bn_history, stage_waits_history = optimize_T3(start_caps_T3)

stages = ["gate", "checkin", "security", "boarding"]

before = [baseline_stats["avg_waits"][s] for s in stages]
after = [final_stats["avg_waits"][s] for s in stages]

plt.figure(figsize=(7,5))
x = np.arange(len(stages))
width = 0.35

plt.bar(x - width/2, before, width, label='Before', alpha=0.8)
plt.bar(x + width/2, after, width, label='After', alpha=0.8)

plt.xticks(x, ["Gate", "Check-in", "Security", "Boarding"])
plt.ylabel("Average Waiting Time (min)")
plt.title("Stage-Wise Average Waiting Time (Before vs After)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(6,4))
plt.bar(["Before", "After"], [baseline_stats["avg_waits"]["total"], final_stats["avg_waits"]["total"]],
        color=["salmon","seagreen"])
plt.ylabel("Total Waiting Time (min)")
plt.title("Reduction in Total Waiting Time")
plt.grid(True, alpha=0.3)
plt.show()

labels = ["Gate Reg", "Gate Digi", "Check-in", "Security"]
before_caps = [start_caps_T3["gate_reg"], start_caps_T3["gate_digi"], start_caps_T3["checkin"], start_caps_T3["security"]]
after_caps  = [best_caps["gate_reg"], best_caps["gate_digi"], best_caps["checkin"], best_caps["security"]]

plt.figure(figsize=(7,5))
x = np.arange(len(labels))
plt.bar(x-0.2, before_caps, width=0.4, label="Before", alpha=0.8)
plt.bar(x+0.2, after_caps,  width=0.4, label="After", alpha=0.8)

plt.xticks(x, labels)
plt.ylabel("Number of Counters")
plt.title("Increase in Service Counters (Before vs After)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

def plot_bottleneck_shift(history_caps, history_stats):
    """
    history_caps: list of caps dict after each accepted step
    history_stats: list of avg_waits dict after each accepted step
    """
    bottlenecks = []
    for s in history_stats:
        waits = s["avg_waits"]
        bn = max(
            {"Gate":waits["gate"],"Check-in":waits["checkin"],
             "Security":waits["security"],"Boarding":waits["boarding"]},
            key=lambda k: {"Gate":waits["gate"],"Check-in":waits["checkin"],
                           "Security":waits["security"],"Boarding":waits["boarding"]}[k]
        )
        bottlenecks.append(bn)

    plt.figure(figsize=(8,4))
    plt.plot(bottlenecks, '-o')
    plt.title("Bottleneck Shift Across Optimization Steps")
    plt.xlabel("Accepted Step #")
    plt.ylabel("Dominant Bottleneck Stage")
    plt.grid(True, alpha=0.3)
    plt.show()

plot_bottleneck_shift(history_caps, history_stats)

before_stats = _run_avg_t3(start_caps_T3)
after_stats  = best_stats   # already returned from optimize_T3()

def plot_before_after_bottleneck(before_stats, after_stats):
    before = before_stats["avg_waits"]
    after = after_stats["avg_waits"]

    labels = ["Gate", "Check-in", "Security", "Boarding"]

    before_vals = [before["gate"], before["checkin"], before["security"], before["boarding"]]
    after_vals  = [after["gate"],  after["checkin"],  after["security"],  after["boarding"]]

    fig, axes = plt.subplots(1, 2, figsize=(14,6))

    # ---- BEFORE PLOT ----
    axes[0].pie(before_vals, labels=labels, autopct='%1.1f%%', startangle=90, explode=[0,0,0,0])
    axes[0].set_title("Before Optimization")

    # ---- AFTER PLOT ----
    axes[1].pie(after_vals, labels=labels, autopct='%1.1f%%', startangle=90, explode=[0,0,0,0])
    axes[1].set_title("After Optimization")

    plt.suptitle("T3 Bottleneck Stage Share Comparison (Before vs After)", fontsize=16)
    plt.show()

plot_before_after_bottleneck(before_stats, after_stats)

# ===============================================
# EXTRACT ALL DATA FROM history_caps & history_stats
# AND EXPORT TO EXCEL
# ===============================================

def extract_full_history(history_caps, history_stats, save_excel=True, filename="T3_Optimization_Full_Data.xlsx"):

    records = []
    for step, (caps, stats) in enumerate(zip(history_caps, history_stats)):
        w = stats["avg_waits"]
        bn = max(
            {"Gate": w["gate"], "Check-in": w["checkin"], "Security": w["security"], "Boarding": w["boarding"]},
            key=lambda k: {"Gate": w["gate"], "Check-in": w["checkin"], "Security": w["security"], "Boarding": w["boarding"]}[k]
        )

        records.append({
            "Step": step,
            # capacities
            "gate_reg": caps["gate_reg"],
            "gate_digi": caps["gate_digi"],
            "checkin": caps["checkin"],
            "security": caps["security"],
            "num_gates": caps["num_gates"],

            # waits
            "wait_gate": w["gate"],
            "wait_checkin": w["checkin"],
            "wait_security": w["security"],
            "wait_boarding": w["boarding"],
            "total_wait": w["total"],

            # bottleneck
            "bottleneck": bn
        })

    df = pd.DataFrame(records)

    if save_excel:
        df.to_excel(filename, index=False)
        print(f"✅ Full optimization dataset saved to:  {filename}")

    return df


# ---- RUN EXTRACTION ----
full_history_df = extract_full_history(history_caps, history_stats)

full_history_df.head()

import pandas as pd
import matplotlib.pyplot as plt

# ===== LOAD DATA =====
# T1
t1_hist = pd.read_excel("data/Optimization_Results_T1 (2).xlsx", sheet_name="history")
t1_base = t1_hist.iloc[0]["total_avg_wait"]
t1_best = t1_hist.iloc[-1]["total_avg_wait"]

# T2
t2_hist = pd.read_excel("data/Optimization_Results_T2 (1).xlsx", sheet_name="history")
t2_base = t2_hist.iloc[0]["total_avg_wait"]
t2_best = t2_hist.iloc[-1]["total_avg_wait"]

# T3
t3_hist = pd.read_excel("data/T3_Optimization_Full_Data.xlsx")
t3_hist["total_avg_wait"] = (t3_hist["wait_gate"] +
                             t3_hist["wait_checkin"] +
                             t3_hist["wait_security"] +
                             t3_hist["wait_boarding"])
t3_base = t3_hist.iloc[0]["total_avg_wait"]
t3_best = t3_hist.iloc[-1]["total_avg_wait"]

# ===== PREPARE DATA =====
terminals = ["T1", "T2", "T3"]
before = [t1_base, t2_base, t3_base]
after = [t1_best, t2_best, t3_best]

x = range(len(terminals))
width = 0.35  # bar width

# ===== PLOT =====
plt.figure(figsize=(8,5))
plt.bar([p - width/2 for p in x], before, width=width, label="Before Optimization", alpha=0.85)
plt.bar([p + width/2 for p in x], after, width=width, label="After Optimization", alpha=0.85)

plt.xticks(x, terminals, fontsize=12)
plt.ylabel("Total Average Waiting Time (min)", fontsize=12)
plt.title("Comparison of Passenger Waiting Times Before and After Optimization", fontsize=14, weight="bold")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

