"""
Semaine 3 — Preference learning, premiere version.
Un occupant a une temperature preferee CACHEE. On ne voit que ses reactions
("trop froid" / "c'est bon" / "trop chaud"). Un estimateur bayesien
doit deviner la preference cachee a partir de ces seules reactions.
"""
import numpy as np
from math import erf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

def Phi(x):                      # fonction de repartition de la loi normale
    return 0.5 * (1 + np.vectorize(erf)(x / np.sqrt(2)))

# ---------- 1) L'OCCUPANT SYNTHETIQUE (verite cachee) ----------
P_TRUE = 25.0                    # sa temperature preferee (l'estimateur l'ignore)
W      = 1.0                     # demi-largeur de sa zone de confort (deg C)
SIGMA  = 2.0                     # bruit : il n'est pas parfaitement coherent

def occupant_feedback(T):
    """Reaction de l'occupant a une temperature T (avec un peu de hasard)."""
    perceived = (T - P_TRUE) + rng.normal(0, SIGMA)
    if perceived >  W: return "too_hot"
    if perceived < -W: return "too_cold"
    return "confortable"

# ---------- 2) L'ESTIMATEUR BAYESIEN ----------
grid = np.arange(16, 28.01, 0.25)     # toutes les preferences candidates
log_belief = np.zeros_like(grid)      # croyance (en log), uniforme au depart

def likelihood(T, fb):
    """Probabilite de la reaction 'fb' a la temperature T, pour CHAQUE candidate."""
    diff = T - grid
    p_hot  = 1 - Phi((W - diff) / SIGMA)
    p_cold = Phi((-W - diff) / SIGMA)
    p_comf = np.clip(1 - p_hot - p_cold, 1e-9, None)
    if fb == "too_hot": return np.clip(p_hot, 1e-9, None)
    if fb == "too_cold": return np.clip(p_cold, 1e-9, None)
    return p_comf

# ---------- 3) L'EXPERIENCE ----------
# Un "controleur" fait varier la temperature de la piece pour explorer (marche aleatoire).
T = 22.0
N = 1000
estimates, snapshots = [], {}
for k in range(N):
    T = np.clip(T + rng.normal(0, 0.8), 18, 26)
    fb = occupant_feedback(T)
    log_belief += np.log(likelihood(T, fb))            # <-- mise a jour bayesienne
    belief = np.exp(log_belief - log_belief.max())
    belief /= belief.sum()
    estimates.append((belief * grid).sum())            # estimation = moyenne de la croyance
    if k + 1 in (5, 25, 200):
        snapshots[k + 1] = belief.copy()

print(f"TRUE Preference (disabled) : {P_TRUE:.2f} C")
print(f"ESTIMED Preference after {N} reactions : {estimates[-1]:.2f} C")

# ---------- 4) VISUALISATION ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

for n_obs, bel in snapshots.items():
    ax1.plot(grid, bel, label=f"after {n_obs} reactions")
ax1.axvline(P_TRUE, color="k", ls="--", label="true preference")
ax1.set_xlabel("Preferential candidate temperature ( C)")
ax1.set_ylabel("Belief (probability)")
ax1.set_title("The belief concentrates on the truth")
ax1.legend()

ax2.plot(range(1, N + 1), estimates, color="#D85A30", label="estimation")
ax2.axhline(P_TRUE, color="k", ls="--", label="true preference")
ax2.set_xlabel("Number of observed reactions")
ax2.set_ylabel("Estimated Preference ( C)")
ax2.set_title("Convergence towards the true preference")
ax2.legend()

plt.tight_layout()
plt.savefig("preference_convergence.png", dpi=110)
print("Figure enregistree.")
