# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 11:40:59 2025

@author: gor38rv
"""

from typing import Optional
import numpy as np
import pandas as pd
import gymnasium as gym
from collections import OrderedDict

# ---- Load data
df = pd.read_csv("hopefully_final_summer_dataset.csv")
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

# ---- Simple thermal helpers (use env.dt_hours)
R, C = 2.2, 3.5      # K/kW, kWh/K
Pmax_cool = 3.0      # kW
Pmax_heat = 4.0      # kW
k_cool = 1.0
k_heat = 1.0
Q_int_active, Q_int_idle = 0.0, 0.0

def Q_internal(mode_str: Optional[str]) -> float:
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

def update_Tin(Tin, Tout, setpoint, dt_hours, occ_mode_str=None, hvac_mode="AUTO"):
    q_env  = (Tout - Tin) / R
    q_hvac = Q_HVAC_both(Tin, setpoint, hvac_mode)
    q_int  = Q_internal(occ_mode_str)
    return Tin + dt_hours / C * (q_env + q_hvac + q_int)

def _cop_cool(Tout: float) -> float:
    return float(np.clip(3.2 - 0.05 * (Tout - 25.0), 1.5, 4.5))

def _cop_heat(Tout: float) -> float:
    # ~3.0 at 10°C, degrades when very cold; clipped to [1.8, 4.0]
    return float(np.clip(3.0 - 0.02 * (10.0 - Tout), 1.8, 4.0))

_FAN_ACTIVE_KW = 0.10
_FAN_STANDBY_KW = 0.03

def _step_elec_kwh(q_hvac_kw: float, Tout: float, dt_hours: float) -> float:
    """Map delivered thermal kW (+heat, -cool) to electrical kWh for this step,
    adding fan/standby draw."""
    if q_hvac_kw > 0.0:      # heating
        comp_kw = q_hvac_kw / max(1e-6, _cop_heat(Tout))
        elec_kw = comp_kw + _FAN_ACTIVE_KW
    elif q_hvac_kw < 0.0:    # cooling
        comp_kw = (-q_hvac_kw) / max(1e-6, _cop_cool(Tout))
        elec_kw = comp_kw + _FAN_ACTIVE_KW
    else:
        elec_kw = _FAN_STANDBY_KW
    return float(elec_kw * dt_hours)

class HVAC(gym.Env):
    def __init__(self,
                 temp_min_out: float = -20,
                 temp_max_out: float = 45,
                 temp_min_in: float = 10,
                 temp_max_in: float = 35):
        super().__init__()

        self.dt_hours = 1.0 / 6.0            # 10 minutes
        self.steps_total = 143            # 144 steps
        self._step_count = 0

        self.setpoints = np.array([18, 19, 20, 21, 22, 23, 24, 25, 26], dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(self.setpoints))

        self.observation_space = gym.spaces.Dict(OrderedDict([
            ("T_out", gym.spaces.Box(low=np.float32(temp_min_out), high=np.float32(temp_max_out), shape=(), dtype=np.float32)),
            ("T_in",  gym.spaces.Box(low=np.float32(temp_min_in),  high=np.float32(temp_max_in),  shape=(), dtype=np.float32)),
            ("is_weekend", gym.spaces.Discrete(2)),
            ("time", gym.spaces.Box(low=np.float32(0.0), high=np.float32(24.0), shape=(), dtype=np.float32)),
            ("prev_action", gym.spaces.Discrete(len(self.setpoints))),
            #("energy_term", gym.spaces.Box(low = np.float32(-100), high = np.float32(0), shape=(), dtype = np.float32)),
            ("elec_kwh", gym.spaces.Box(low=np.float32(0.0), high=np.float32(5.0), shape=(), dtype=np.float32)),
        ]))

        # internal state
        self._T_out = np.float32(0.0)
        self._T_in = np.float32(0.0)
        self._is_weekend = 0
        self._time = np.float32(0.0)      # hours since midnight
        self._prev_action = 0             # index 0..3
        self._date = None                 # python date
        self._daily_df = None             # cached day slice
        #self._energy_term = 0
        # --- ADDED: last-step electrical energy (kWh)
        self._elec_kwh = np.float32(0.0)

    def _get_obs(self):
        return {
            "T_out": self._T_out,
            "T_in": self._T_in,
            "is_weekend": int(self._is_weekend),
            "time": self._time,
            "prev_action": int(self._prev_action), 
            #"energy_term": float(self._energy_term),
            "elec_kwh": float(self._elec_kwh),
        }

    def _get_info(self):
        t = self._step_count * self.dt_hours               # 0.0 .. 23.8333
        steps_remaining = self.steps_total - self._step_count
        return {
            "time_of_day": np.float32(t),
            "hours_remaining": np.float32(24.0 - t),
            "steps_elapsed": int(self._step_count),
            "steps_remaining": int(steps_remaining),
            "episode_progress": t / 24.0
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._step_count = 0

        # choose a random day, then take the 00:00 row for that day
        day = df["time"].dt.date.sample(n=1, random_state=seed).iloc[0]
        self._date = day
        self._daily_df = df.loc[df["time"].dt.date == day].sort_values("time").reset_index(drop=True)

        # find the first row (assumes you have a 10-min grid with an entry at 00:00)
        row0 = self._daily_df.iloc[0]

        # prev_action: map previous setpoint value to index (fallback to nearest)
        prev_val = float(row0["prev_setpoint_C"])
        idxs = np.where(np.isclose(self.setpoints, prev_val))[0]
        prev_idx = int(idxs[0]) if len(idxs) else int(np.argmin(np.abs(self.setpoints - prev_val)))

        self._T_out = np.float32(row0["Tout_C"])
        self._T_in = np.float32(row0["Tin_C"])
        self._is_weekend = int(row0["weekend"])
        self._time = np.float32(0.0)      # start-of-day
        self._prev_action = prev_idx
        a_prev_c = float(self.setpoints[self._prev_action])
        #self._energy_term = -abs((a_prev_c - float(self._T_out)) + (a_prev_c - float(self._T_in)))
        # --- ADDED:
        self._elec_kwh = np.float32(0.0)
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        # map action index -> setpoint value
        setpoint = float(self.setpoints[int(action)])

        # --- ADDED: compute HVAC thermal power *before* updating Tin (same inputs as update_Tin)
        q_hvac_kw = Q_HVAC_both(float(self._T_in), setpoint, "AUTO")
        # per-step electrical energy (kWh) from thermal power and Tout
        elec_kwh = _step_elec_kwh(q_hvac_kw, float(self._T_out), float(self.dt_hours))

        # advance step counter first
        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.steps_total
        
        idx = min(self._step_count, len(self._daily_df) - 1)
        tout_next = float(self._daily_df.iloc[idx]["Tout_C"])
        act_map = {20:0, 22:1, 24:2, 26:3}
        # evolve Tin (unchanged dynamics)
        self._T_in = np.float32(update_Tin(self._T_in, self._T_out, setpoint, self.dt_hours))
        self._T_out = np.float32(tout_next)
        self._time = np.float32(self._step_count * self.dt_hours)   # derived to avoid drift
        self._prev_action = int(action)
        a_prev_c = float(self.setpoints[self._prev_action])
        #self._energy_term = -abs((a_prev_c - float(self._T_out)) + (a_prev_c - float(self._T_in)))
        # --- ADDED: expose per-step kWh
        self._elec_kwh = np.float32(elec_kwh)

        reward = 0.0

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), self._get_info()
