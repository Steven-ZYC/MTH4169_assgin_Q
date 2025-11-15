# -*- coding: utf-8 -*-
"""
English-only figure generator for the Russian Roulette paper.

Outputs six figures:
(1) Theoretical vs Monte Carlo p_i comparison
(2) Fairness index Φ(C,n,m) heatmap
(3) Two-player advantage Δ = p1 - p2 map
(4) V_k curve with optimal stopping point k*
(5) Optimal stopping regions on (C,n) plane
(6) k* trajectories under varying U_w and U_d

Rules:
- Use matplotlib only (no seaborn).
- One chart per figure (no subplots).
- Do not set specific colors/styles.
- Save figures under /mnt/data as PNG.

"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import logging
import time

# ensure output folder exists (place images next to this script)
script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
out_dir = os.path.join(script_dir, "img")
os.makedirs(out_dir, exist_ok=True)

# --------------------------
# Logging helpers (added): terminal + file with timestamp
# --------------------------

logs_dir = os.path.join(script_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)

def setup_logger(name: str = "montecarlo", level=logging.INFO):
    """Create a logger that writes to terminal and a timestamped file."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(logs_dir, f"{name}-{ts}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.propagate = False
    logger.info(f"Log file initialized: {log_path}")
    return logger, log_path

# Global logger
LOGGER, LOG_PATH = setup_logger("montecarlo")

# --------------------------
# Core probability / fairness / simulation / DP helpers
# --------------------------

def death_probs_theory(C: int, n: int, m: int):
    """
    Theoretical death probabilities for one-shot, no-spin variant.
    Player order is fixed; player i shoots the (i-th) chamber exactly once.
    p_i = [∏_{j=1}^{i-1} (C-j+1-n)/(C-j+1)] * [n/(C-i+1)]
    """
    probs = []
    survive_prefix = 1.0
    for i in range(1, m + 1):
        p_i = survive_prefix * (n / (C - i + 1))
        probs.append(p_i)
        survive_prefix *= (C - i + 1 - n) / (C - i + 1)
    return probs


def simulate_one_shot(
    C: int,
    n: int,
    m: int,
    trials: int = 200_000,
    rng_seed: int = 42,
    logger: logging.Logger = None,
    log_every: int = 10_000
):
    """
    Monte Carlo for one-shot, no-spin:
    - Randomly place n bullets among C chambers (without replacement).
    - Players 1..m each take one shot in order.
    - If a player hits a bullet, that player dies and the round ends.
    Returns empirical death probabilities for players 1..m.

    Added behavior:
    - Periodically logs progress and running estimates to terminal and to a timestamped log file.
    - On KeyboardInterrupt, logs partial estimates before returning.
    """
    if logger is None:
        logger = LOGGER

    rnd = random.Random(rng_seed)
    deaths = [0] * m
    start = time.perf_counter()

    logger.info(f"[MC-START] C={C}, n={n}, m={m}, trials={trials}, seed={rng_seed}")
    logger.info(f"[MC-START] log_every={log_every}")

    try:
        for t in range(trials):
            bullet_positions = set(rnd.sample(range(C), n))
            for i in range(m):
                if i in bullet_positions:
                    deaths[i] += 1
                    break

            # Progress logging
            step = t + 1
            if step % log_every == 0 or step == trials:
                running_probs = [d / step for d in deaths]
                elapsed = time.perf_counter() - start
                pct = 100.0 * step / trials
                logger.info(
                    f"[MC-PROGRESS] {step}/{trials} ({pct:.1f}%) "
                    f"elapsed={elapsed:.2f}s "
                    f"running p_i={['{:.6f}'.format(x) for x in running_probs]}"
                )

    except KeyboardInterrupt:
        # Record partial results if interrupted
        step = max(1, t + 1)  # t may be undefined in some edge cases
        running_probs = [d / step for d in deaths]
        elapsed = time.perf_counter() - start
        logger.warning(
            f"[MC-INTERRUPTED] at {step}/{trials} "
            f"elapsed={elapsed:.2f}s "
            f"partial p_i={['{:.6f}'.format(x) for x in running_probs]}"
        )

    final_probs = [d / trials for d in deaths]
    total_elapsed = time.perf_counter() - start
    logger.info(f"[MC-DONE] total_elapsed={total_elapsed:.2f}s final p_i={['{:.6f}'.format(x) for x in final_probs]}")
    return final_probs


def fairness_index_from_probs(ps):
    """Standard deviation of death probabilities."""
    ps = np.array(ps, dtype=float)
    return float(ps.std())


def fairness_index(C: int, n: int, m: int):
    """Compute Φ from theory p_i."""
    ps = death_probs_theory(C, n, m)
    return fairness_index_from_probs(ps)


def advantage_gap_two_players(C: int, n: int):
    """
    Δ = p1 - p2 for m=2.
    p1 = n/C
    p2 = (C-n)/C * n/(C-1) = n(C-n)/(C(C-1))
    Δ = (n/C) - n(C-n)/(C(C-1)) = (n/C) * ((n-1)/(C-1))
    """
    if C <= 1 or n < 0 or n >= C:
        return None
    p1 = n / C
    p2 = (C - n) / C * (n / (C - 1))
    return p1 - p2


def Vk_curve(C: int, n: int, Uwin: float = 10.0, Uforfeit: float = 0.0, Udeath: float = -1.0):
    """
    Dynamic programming for optimal stopping (two players, no spin, fixed layout per round):
    Round k has two shots: you first, then opponent if you survive.
    V_k = P_die*Udeath + P_win*Uwin + P_continue*V_{k+1}
    Terminal: for the last feasible round k_max, if d_k = n (only bullets remain), set V_k = Udeath.
    Returns: list of (k, V_k) for k=1..k_max, and k_star = max k with V_k > Uforfeit (0 if none).
    """
    # Find k_max such that d_k = C - 2k + 2 >= n and at least 2 chambers remain.
    kmax = 0
    k = 1
    while True:
        d_k = C - 2 * k + 2
        if d_k >= n and d_k >= 2:
            kmax = k
            k += 1
        else:
            break
    if kmax == 0:
        return [], 0

    V = {}
    d_kmax = C - 2 * kmax + 2
    if d_kmax == n:
        V[kmax] = Udeath
    else:
        d = d_kmax
        P_die = n / d
        P_win = (d - n) / d * (n / (d - 1)) if d > 1 else 0.0
        P_cont = (d - n) / d * ((d - 1 - n) / (d - 1)) if d > 1 else 0.0
        V[kmax] = P_die * Udeath + P_win * Uwin + P_cont * Uforfeit

    for k in range(kmax - 1, 0, -1):
        d = C - 2 * k + 2
        P_die = n / d
        if d > 1:
            P_win = (d - n) / d * (n / (d - 1))
            P_cont = (d - n) / d * ((d - 1 - n) / (d - 1))
        else:
            P_win = 0.0
            P_cont = 0.0
        V[k] = P_die * Udeath + P_win * Uwin + P_cont * V[k + 1]

    k_star = 0
    for k in range(1, kmax + 1):
        if V[k] > Uforfeit:
            k_star = k

    curve = [(k, V[k]) for k in range(1, kmax + 1)]
    return curve, k_star


# --------------------------
# Figure 1: Theory vs Monte Carlo p_i
# --------------------------
def figure1_pi_compare(C=6, n=2, m=3, trials=150_000): 
    # Fig. 1 now: Two-player advantage heatmap (previously Fig. 3)
    Cs = list(range(3, 21))
    max_n = 20 - 1
    delta_grid = np.full((len(Cs), max_n), np.nan, dtype=float)

    for i, Cc in enumerate(Cs):
        for nn in range(1, Cc):
            delta_grid[i, nn - 1] = advantage_gap_two_players(Cc, nn)

    plt.figure(figsize=(7, 5))
    plt.imshow(delta_grid, aspect='auto', origin='lower')
    plt.colorbar(label='$\\Delta = p_1 - p_2$')
    plt.title('Fig. 1 — Two-Player Advantage Map $\\Delta(C,n)$')
    plt.xlabel('Bullets n')
    plt.ylabel('Chambers C')
    plt.xticks(range(max_n), range(1, max_n + 1))
    plt.yticks(range(len(Cs)), Cs)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig1_advantage_heatmap_en.png'), dpi=200)
    plt.close()


# --------------------------
# Figure 2: Theoretical vs Monte Carlo p_i (swapped)
# --------------------------
def figure2_phi_heatmap(m=2, C_min=3, C_max=12):
    # Fig. 2 now: Theoretical vs Monte Carlo $p_i$ (previously Fig. 3 content)
    C = 6
    n = 2
    m_local = 3
    trials = 150_000
    theory = death_probs_theory(C, n, m_local)

    # With logging-enabled Monte Carlo (added)
    log_every = max(1_000, trials // 20)  # print every ~5%
    LOGGER.info(f"[FIG2] Running MC compare: C={C}, n={n}, m={m_local}, trials={trials}")
    sim = simulate_one_shot(
        C, n, m_local,
        trials=trials,
        rng_seed=123,
        logger=LOGGER,
        log_every=log_every
    )

    players = np.arange(1, m_local + 1)
    width = 0.35

    plt.figure(figsize=(7, 4.5))
    plt.bar(players - width/2, theory, width, label='Theory')
    plt.bar(players + width/2, sim, width, label='Monte Carlo')
    plt.title(f'Fig. 2 — Theoretical vs Monte Carlo $p_i$  |  C={C}, n={n}, m={m_local}, trials={trials}')
    plt.xlabel('Player index i')
    plt.ylabel('Death probability $p_i$')
    plt.xticks(players)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig2_pi_compare_en.png'), dpi=200)
    plt.close()

    LOGGER.info("[FIG2] Saved fig2_pi_compare_en.png")


# --------------------------
# Figure 3: Fairness index heatmap Φ(C,n,m) (swapped)
# --------------------------
def figure3_advantage_heatmap(C_min=3, C_max=20):
    # Fig. 3 now: Fairness Index heatmap (previously Fig. 2 content)
    Cs = list(range(C_min, C_max + 1))
    max_n = C_max - 1
    phi_grid = np.full((len(Cs), max_n), np.nan, dtype=float)

    for i, Cc in enumerate(Cs):
        for nn in range(1, Cc):
            phi_grid[i, nn - 1] = fairness_index(Cc, nn, m=2)

    plt.figure(figsize=(7, 5))
    plt.imshow(phi_grid, aspect='auto', origin='lower')
    plt.colorbar(label='Fairness index $\\Phi$')
    plt.title('Fig. 3 — Fairness Index $\\Phi(C,n,m)$ Heatmap  |  m=2')
    plt.xlabel('Bullets n')
    plt.ylabel('Chambers C')
    plt.xticks(range(max_n), range(1, max_n + 1))
    plt.yticks(range(len(Cs)), Cs)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig3_phi_heatmap_en.png'), dpi=200)
    plt.close()


# --------------------------
# Figure 4: V_k curve with optimal stopping point k*
# --------------------------
def figure4_Vk_curve(C=6, n=2, Uwin=10, Udeath=-1, Uforfeit=0):
    curve, k_star = Vk_curve(C, n, Uwin, Uforfeit, Udeath)
    ks = [k for k, _ in curve]
    Vs = [v for _, v in curve]

    plt.figure(figsize=(7, 4.5))
    plt.plot(ks, Vs, marker='o')
    plt.axhline(0, linestyle='--')
    if k_star > 0:
        v_star = dict(curve)[k_star]
        plt.scatter([k_star], [v_star], s=80)
        plt.text(k_star, v_star, f'  k*={k_star}', va='bottom')
    plt.title(f'Fig. 4 — $V_k$ Curve and Optimal Stopping  |  C={C}, n={n}, Uw={Uwin}, Ud={Udeath}')
    plt.xlabel('Round k')
    plt.ylabel('Expected utility $V_k$')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig4_Vk_curve_en.png'), dpi=200)
    plt.close()


# --------------------------
# Figure 5: Optimal stopping regions on (C,n) plane
# --------------------------
def figure5_stopping_regions(C_min=4, C_max=16, Uwin=10, Udeath=-1, Uforfeit=0):
    Cs = list(range(C_min, C_max + 1))
    max_n = C_max - 1
    # Region code: 0 = immediate forfeit (V1<=0), 1 = early stop, 2 = late stop, 3 = never stop until kmax
    region = np.full((len(Cs), max_n), np.nan, dtype=float)

    for i, C in enumerate(Cs):
        for n in range(1, C):
            curve, k_star = Vk_curve(C, n, Uwin, Uforfeit, Udeath)
            if not curve:
                region[i, n - 1] = 0
                continue
            kmax = curve[-1][0]
            if k_star == 0:
                region[i, n - 1] = 0
            elif k_star == kmax:
                region[i, n - 1] = 3
            else:
                ratio = k_star / kmax if kmax > 0 else 0
                region[i, n - 1] = 1 if ratio < 0.4 else 2

    plt.figure(figsize=(7, 5))
    plt.imshow(region, aspect='auto', origin='lower')
    cbar = plt.colorbar()
    cbar.set_label('Stopping region (0=immediate, 1=early, 2=late, 3=never)')
    plt.title(f'Fig. 5 — Optimal Stopping Regions on (C,n)  |  Uw={Uwin}, Ud={Udeath}')
    plt.xlabel('Bullets n')
    plt.ylabel('Chambers C')
    plt.xticks(range(max_n), range(1, max_n + 1))
    plt.yticks(range(len(Cs)), Cs)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig5_stopping_regions_en.png'), dpi=200)
    plt.close()


# --------------------------
# Figure 6: k* trajectories under varying U_w and U_d
# --------------------------
def figure6_kstar_trajectories(C=10, n=3):
    # Trajectory 1: vary U_w with fixed U_d = -1
    Uw_values = np.arange(2, 21, 1)  # 2..20
    kstar_vs_Uw = []
    for Uw in Uw_values:
        curve, k_star = Vk_curve(C, n, Uwin=Uw, Uforfeit=0.0, Udeath=-1.0)
        kstar_vs_Uw.append(k_star)

    # Trajectory 2: vary U_d with fixed U_w = 10
    Ud_values = np.arange(-10, 0, 1)  # -10..-1
    kstar_vs_Ud = []
    for Ud in Ud_values:
        curve, k_star = Vk_curve(C, n, Uwin=10.0, Uforfeit=0.0, Udeath=Ud)
        kstar_vs_Ud.append(k_star)

    plt.figure(figsize=(7, 4.5))
    plt.plot(Uw_values, kstar_vs_Uw, marker='o', label='k* vs U_w (U_d = -1)')
    plt.plot(Ud_values, kstar_vs_Ud, marker='s', label='k* vs U_d (U_w = 10)')
    plt.title(f'Fig. 6 — $k^*$ Trajectories vs $U_w$ and $U_d$  |  C={C}, n={n}')
    plt.xlabel('Utility parameter (left: U_w, right: U_d)')
    plt.ylabel('Optimal stopping round $k^*$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig6_kstar_trajectories_en.png'), dpi=200)
    plt.close()


# --------------------------
# Generate all six figures with default parameters
# --------------------------

if __name__ == "__main__":
    figure1_pi_compare(C=6, n=2, m=3, trials=120_000)
    figure2_phi_heatmap(m=2, C_min=3, C_max=12)
    figure3_advantage_heatmap(C_min=3, C_max=20)
    figure4_Vk_curve(C=6, n=2, Uwin=10, Udeath=-1, Uforfeit=0)
    figure5_stopping_regions(C_min=4, C_max=16, Uwin=10, Udeath=-1, Uforfeit=0)
    figure6_kstar_trajectories(C=10, n=3)

    print(f"Saved: {os.path.join(out_dir, 'fig1_advantage_heatmap_en.png')}")
    print(f"Saved: {os.path.join(out_dir, 'fig2_pi_compare_en.png')}")
    print(f"Saved: {os.path.join(out_dir, 'fig3_phi_heatmap_en.png')}")
    print(f"Saved: {os.path.join(out_dir, 'fig4_Vk_curve_en.png')}")
    print(f"Saved: {os.path.join(out_dir, 'fig5_stopping_regions_en.png')}")
    print(f"Saved: {os.path.join(out_dir, 'fig6_kstar_trajectories_en.png')}")
    print(f"Log file: {LOG_PATH}")
