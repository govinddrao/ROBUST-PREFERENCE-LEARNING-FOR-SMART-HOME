# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 13:44:11 2026

@author: gor38rv
"""

import numpy as np

class HVACExpertOracle:
    """
    Expert oracle compatible with your HVAC Gym env:
    - obs is dict with keys: T_out, T_in, is_weekend, time, prev_action, elec_kwh
    - action returned is an int in {0..8} (index into env.setpoints)

    Privacy: mode is NOT in obs; expert maintains belief b_t over modes using IO-HMM transitions.
    """

    def __init__(self, setpoints, beta=1.0, seed=0):
        self.setpoints = np.asarray(setpoints, dtype=float)  # [18..26]
        self.beta = float(beta)
        self.rng = np.random.default_rng(seed)

        # ----- same as your generator -----
        # modes: away=0, active=1, sleeping=2
        self.num_k = 3
        self.INIT_MODE_PROBS = np.array([0.2, 0.2, 0.6], dtype=float)

        self.ALPHA = np.array([
            [ 1.7,  0.9, 0.5],   # prev=Away
            [ 1.2,  2.0, 0.5],   # prev=Active
            [-1.0,  1.0, 1.5],   # prev=Sleeping
        ], dtype=float)

        self.GAMMA = np.array([
            # night  work  weekend  hot
            [-0.1,  2.3,   0.7,    0.0],   # Away
            [-0.4,  0.1,   0.5,    0.9],   # Active
            [ 2.0, -2.2,   0.99,   0.5],   # Sleeping
        ], dtype=float)

        self.mean_temps = np.array([26., 24., 21.])  # anchors
        self.theta = np.array([1.0, 0.0], dtype=float)
        self.rho_change = 7.2

        eta_away   = np.linspace(-2.5, 2.0, len(self.setpoints))
        eta_active = np.array([-1.0, -0.7, -0.2, 0.5, 1.0, 1.3, 1.5, 0.6, -0.3], dtype=float)
        eta_sleep  = np.array([ 1.6,  1.5,  1.3, 1.1, 0.7, 0.2,-0.4,-1.2,-2.0], dtype=float)
        self.eta = np.vstack([eta_away, eta_active, eta_sleep])

        # internal memory
        self.belief = None     # shape (3,)
        self.a_prev = None     # previous setpoint value (float)

    @staticmethod
    def _softmax(v):
        v = np.asarray(v, dtype=float)
        m = np.max(v)
        e = np.exp(v - m)
        return e / (e.sum() + 1e-12)

    @staticmethod
    def _psi(hour, weekend, Tout):
        night = 1.0 if (hour >= 22 or hour < 7) else 0.0
        work  = 1.0 if (9 <= hour < 17 and not weekend) else 0.0
        hot   = 1.0 if Tout > 28 else 0.0
        return np.array([night, work, weekend, hot], dtype=float)

    def reset(self, obs=None):
        self.belief = self.INIT_MODE_PROBS.copy()
        if obs is not None:
            prev_idx = int(obs["prev_action"])
            self.a_prev = float(self.setpoints[prev_idx])
        else:
            self.a_prev = float(self.rng.choice(self.setpoints))

    def _transition_matrix(self, hour, weekend, Tout):
        """
        Build T_t where T[i,j] = P(mode_t=j | mode_{t-1}=i, psi_t)
        using your ALPHA + GAMMA @ psi
        """
        psi = self._psi(hour, weekend, Tout)
        T = np.zeros((self.num_k, self.num_k), dtype=float)
        g = self.GAMMA @ psi  # shape (3,)
        for i in range(self.num_k):
            logits = self.ALPHA[i] + g
            T[i] = self._softmax(logits)
        return T

    def _phi_matrix(self, Tin, Tout, mode_idx):
        a = self.setpoints
        cols = [
            -np.abs((self.mean_temps[mode_idx] - a) ** 2),
            -np.abs((a - Tout) + (a - Tin)),
        ]
        return np.column_stack(cols)

    def _utility_mode(self, Tin, Tout, mode_idx):
        Phi = self._phi_matrix(Tin, Tout, mode_idx)
        U_base = Phi @ self.theta
        U = U_base + self.eta[mode_idx] - self.rho_change * np.abs(self.setpoints - self.a_prev)
        return U

    def act(self, obs):
        """
        Returns: (action_idx, action_setpoint, probs_over_actions)
        """
        if self.belief is None or self.a_prev is None:
            self.reset(obs)

        Tin = float(obs["T_in"])
        Tout = float(obs["T_out"])
        weekend = int(obs["is_weekend"])
        hour = float(obs["time"])  # you store hours since midnight

        # 1) IO-HMM belief prediction step (no emissions in your setup)
        T = self._transition_matrix(hour, weekend, Tout)
        self.belief = self.belief @ T
        self.belief = self.belief / (self.belief.sum() + 1e-12)

        # 2) expected utility across modes
        U = np.zeros(len(self.setpoints), dtype=float)
        for k in range(self.num_k):
            U += self.belief[k] * self._utility_mode(Tin, Tout, k)

        # 3) sample action
        if self.beta <= 1e-8:
            probs = np.zeros_like(U)
            probs[int(np.argmax(U))] = 1.0
        else:
            probs = self._softmax(U / self.beta)

        a_idx = int(self.rng.choice(len(self.setpoints), p=probs))
        a_set = float(self.setpoints[a_idx])

        # 4) update memory
        self.a_prev = a_set
        return a_idx, a_set, probs
