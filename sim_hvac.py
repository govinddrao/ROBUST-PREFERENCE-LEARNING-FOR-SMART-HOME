import numpy as np
import pandas as pd

def generate_dataset(
    n_episodes: int = 100,
    dt_min: int = 10,
    process_noise_std: float = 0.0,  # °C std added to Tin_next
    beta: float = 1.0,               # softmax temperature; 0 => greedy
    seed: int = 42,
    reset_tin_each_day: bool = True, # start each episode fresh
    tin_init: float = 22.0,
) -> pd.DataFrame:
    # ---- Config derived from args ----
    STEPS = int(24 * 60 / dt_min)
    ACTIONS = np.arange(18., 27., 1.0, dtype=float)  # 18..26
    dt_hr = dt_min / 60.0
    rng = np.random.default_rng(seed)

    # ---- IO-HMM for occupancy ----
    hidden_modes = {"away": 0, "active": 1, "sleeping": 2}
    idx2mode = {v: k for k, v in hidden_modes.items()}
    num_k = len(hidden_modes)
    INIT_MODE_PROBS = np.array([0.2, 0.2, 0.6], dtype=float)

    def softmax(v):
        m = np.max(v)
        e = np.exp(v - m)
        return e / e.sum()

    def psi(hour, weekend, Tout):
        night = 1.0 if (hour >= 22 or hour < 7) else 0.0
        work  = 1.0 if (9 <= hour < 17 and not weekend) else 0.0
        hot   = 1.0 if Tout > 28 else 0.0
        return np.array([night, work, weekend, hot], dtype=float)

    ALPHA = np.array([
        [ 1.7,  0.9, 0.5],   # prev=Away
        [ 1.2,  2.0, 0.5],   # prev=Active
        [-1.0,  1.0, 1.5],   # prev=Sleeping
    ], dtype=float)

    GAMMA = np.array([
        # night  work  weekend  hot
        [-0.1,  2.3,   0.7,    0.0],   # Away
        [-0.4,  0.1,   0.5,    0.9],   # Active
        [ 2.0, -2.2,   0.99,   0.5],   # Sleeping
    ], dtype=float)

    def sample_mode(prev_mode_idx, hour, weekend, Tout):
        p = softmax(ALPHA[prev_mode_idx] + GAMMA @ psi(hour, weekend, Tout))
        return int(rng.choice(num_k, p=p)), p

    # ---- RC thermal + energy ----
    R, C = 2.2, 3.5
    Pmax_cool, Pmax_heat = 3.0, 4.0
    k_cool, k_heat = 1.0, 1.0
    Q_int_active, Q_int_idle = 0.0, 0.0
    FAN_ACTIVE_KW, FAN_STANDBY_KW = 0.10, 0.03

    def cop_cool(Tout):
        return float(np.clip(3.2 - 0.05*(Tout - 25.0), 1.5, 4.5))

    def cop_heat(Tout):
        return float(np.clip(3.0 - 0.02*(10.0 - Tout), 1.8, 4.0))

    def Q_internal(mode_str: str) -> float:
        return Q_int_active if mode_str == "active" else Q_int_idle

    def Q_HVAC_both(Tin, setpoint, mode_switch="AUTO", deadband=0.5):
        heat_gap = max(0.0, setpoint - Tin - deadband/2)
        cool_gap = max(0.0, Tin - setpoint - deadband/2)
        q_heat = min(Pmax_heat, k_heat * heat_gap)
        q_cool = min(Pmax_cool, k_cool * cool_gap)
        if mode_switch == "HEAT":
            return +q_heat
        if mode_switch == "COOL":
            return -q_cool
        # AUTO
        if heat_gap > 0 and cool_gap == 0:
            return +q_heat
        if cool_gap > 0 and heat_gap == 0:
            return -q_cool
        return 0.0

    def step_energy_kwh(q_hvac_kw, Tout):
        if q_hvac_kw > 0.0:
            elec_kw = q_hvac_kw / max(1e-6, cop_heat(Tout))
        elif q_hvac_kw < 0.0:
            elec_kw = (-q_hvac_kw) / max(1e-6, cop_cool(Tout))
        else:
            elec_kw = 0.0
        elec_kw += FAN_ACTIVE_KW if abs(q_hvac_kw) > 1e-6 else FAN_STANDBY_KW
        return elec_kw * dt_hr

    def update_Tin_and_q(Tin, Tout, setpoint, occ_mode_str=None, hvac_mode="AUTO"):
        q_env  = (Tout - Tin) / R
        q_hvac = Q_HVAC_both(Tin, setpoint, hvac_mode)
        q_int  = Q_internal(occ_mode_str)
        Tin_next = Tin + dt_hr/C * (q_env + q_hvac + q_int)
        # Inject process noise here:
        Tin_next += rng.normal(0.0, process_noise_std)
        return Tin_next, q_hvac

    # ---- Expert policy ----
    mean_temps = np.array([26., 24., 21.])  # away, active, sleeping anchors
    theta = np.array([1.0, 0.0], dtype=float)
    rho_change = 7.2
    eta_away   = np.linspace(-2.5, 2.0, len(ACTIONS))
    eta_active = np.array([-1.0, -0.7, -0.2, 0.5, 1.0, 1.3, 1.5, 0.6, -0.3], dtype=float)
    eta_sleep  = np.array([ 1.6,  1.5,  1.3, 1.1, 0.7, 0.2,-0.4,-1.2,-2.0], dtype=float)
    eta = np.vstack([eta_away, eta_active, eta_sleep])

    def phi_matrix(Tin, actions, T_out, mode_idx, price=None):
        actions = np.asarray(actions, float)
        load = np.maximum(0.0, Tin - actions) * (1.0 + 0.05*np.maximum(0.0, T_out - 25.0))
        cols = [
            -np.abs((mean_temps[mode_idx] - actions)**2),
            -np.abs((actions - T_out) + (actions - Tin)),
        ]
        if price is not None:
            cols.append(-(price * load))
        return np.column_stack(cols)

    def action_probs(Tin, T_out, a_prev, mode_idx, beta: float):
        Phi = phi_matrix(Tin, ACTIONS, T_out, mode_idx)
        U_base = Phi @ theta
        U = U_base + eta[mode_idx] - rho_change * np.abs(ACTIONS - a_prev)
        if beta <= 1e-8:
            p = np.zeros_like(U); p[int(np.argmax(U))] = 1.0  # greedy
        else:
            p = softmax(U / beta)
        return U, p

    # ---- Outdoor forcing ----
    def _cos_ease(y0, y1, r):
        r = np.clip(r, 0.0, 1.0)
        return y1 + (y0 - y1) * 0.5*(1 + np.cos(np.pi * r))

    def _smooth_profile(hour):
        h = hour if hour >= 5.5 else hour + 24.0
        if 5.5 <= h < 15.0:
            r = (h - 5.5)/(15.0 - 5.5); y = _cos_ease(-1.0, +1.0, r)
        elif 15.0 <= h < 19.0:
            y = 1.0
        elif 19.0 <= h < 21.0:
            r = (h - 19.0)/(21.0 - 19.0); y = _cos_ease(1.0, 0.3, r)
        elif 21.0 <= h < 24.0:
            r = (h - 21.0)/(24.0 - 21.0); y = _cos_ease(0.3, -0.6, r)
        else:
            h2 = h if h >= 24.0 else h + 24.0
            r = (h2 - 24.0)/(29.5 - 24.0); y = _cos_ease(-0.6, -1.0, r)
        return y

    def generate_day_forcing(start_time: pd.Timestamp):
        hours = np.arange(STEPS) * (dt_min/60.0)
        day_bias = rng.normal(0.0, 0.3)
        day_amp  = 6.0*(1.0 + rng.normal(0.0, 0.03))
        sigma = 0.05; rho = 0.97; eps = 0.0
        Tout = np.empty(STEPS, dtype=float)
        for i, h in enumerate(hours):
            y = _smooth_profile(h)
            eps = rho*eps + rng.normal(0.0, sigma)
            Tout[i] = (24.0 + day_bias) + day_amp*y + eps
        times = pd.date_range(start_time, periods=STEPS, freq=f"{dt_min}min")
        weekend_seq = (times.dayofweek >= 5).astype(int)  # <-- real weekend
        return hours, weekend_seq, Tout, times

    # ---- Generate ----
    rows = []
    base_date = pd.Timestamp("2025-08-01")
    Tin = float(tin_init)

    for ep in range(1, n_episodes + 1):
        if reset_tin_each_day:
            Tin = np.random.choice([18, 19, 20, 21, 22, 23, 24, 25])  # small spread

        a_prev = float(rng.choice(ACTIONS))
        mode_idx = int(rng.choice(num_k, p=INIT_MODE_PROBS))

        hours, weekend_seq, Tout_seq, times = generate_day_forcing(base_date + pd.Timedelta(days=ep-1))

        for t in range(STEPS):
            h = float(hours[t])
            Tout = float(Tout_seq[t])
            weekend = int(weekend_seq[t])

            # IO-HMM step
            mode_idx, p_modes = sample_mode(mode_idx, h, weekend, Tout)
            mode_str = idx2mode[mode_idx]

            # expert action
            U, p_act = action_probs(Tin, Tout, a_prev, mode_idx, beta=beta)
            a_t = float(rng.choice(ACTIONS, p=p_act))

            # physics + energy
            Tin_next, q_hvac_kw = update_Tin_and_q(Tin, Tout, a_t, mode_str, hvac_mode="AUTO")
            elec_kwh = step_energy_kwh(q_hvac_kw, Tout)
            elec_kw  = elec_kwh / dt_hr

            # diagnostics
            energy_term  = -abs((a_prev - Tout) + (a_prev - Tin))
            comfort_term = -abs(mean_temps[mode_idx] - a_prev)

            row = {
                "episode": ep,
                "step": t,
                "time": times[t],
                "hour": h,
                "weekend": weekend,             # now consistent with sampling
                "Tout_C": Tout,
                "Tin_C": float(Tin),
                "mode": mode_str,
                "mode_p_away": float(p_modes[0]),
                "mode_p_active": float(p_modes[1]),
                "mode_p_sleeping": float(p_modes[2]),
                "setpoint_C": a_t,
                "prev_setpoint_C": float(a_prev),
                "load_balance_term": float(energy_term),
                "comfort_term": float(comfort_term),
                "next_Tin_C": float(Tin_next),
                "hvac_thermal_kW": float(q_hvac_kw),
                "hvac_elec_kW": float(elec_kw),
                "hvac_elec_kWh": float(elec_kwh),
            }
            for i, a in enumerate(ACTIONS.astype(int)):
                row[f"U_{a}"] = float(U[i])
            rows.append(row)

            Tin = Tin_next
            a_prev = a_t

    df = pd.DataFrame(rows)
    # weekend column already correct; keep as int16 if you like
    df["weekend"] = df["weekend"].astype(np.int16)

    print(f"Wrote {len(df)} rows = {n_episodes} × {STEPS} steps")
    print("Columns include hvac_thermal_kW, hvac_elec_kW, hvac_elec_kWh and U_18..U_26.")
    return df
