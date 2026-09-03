"""
Cloture semaine 5 : la vitesse d'oubli OPTIMALE depend-elle de la
vitesse a laquelle la preference change ?
Hypothese : un occupant qui change vite d'avis reclame un oubli plus fort.
"""
import numpy as np
from math import erf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def Phi(x): return 0.5 * (1 + np.vectorize(erf)(x / np.sqrt(2)))
grid = np.arange(16, 30.001, 0.1); dx = grid[1] - grid[0]
W, SIGMA = 1.0, 1.5

def noyau(sd):
    d = int(np.ceil(3 * sd / dx)); x = np.arange(-d, d + 1) * dx
    k = np.exp(-0.5 * (x / sd) ** 2); return k / k.sum()

def vraisemblance(T, fb):
    diff = T - grid
    ph = 1 - Phi((W - diff) / SIGMA); pc = Phi((-W - diff) / SIGMA)
    po = np.clip(1 - ph - pc, 1e-9, None)
    return np.clip({"chaud": ph, "froid": pc, "ok": po}[fb], 1e-9, None)

def erreur_suivi(periode, sd, N=2500, seed=0):
    rng = np.random.default_rng(seed)
    bel = np.ones_like(grid) / len(grid)
    k = noyau(sd) if sd > 0 else None
    errs = []
    for t in range(N):
        p = 23.0 + 2.0 * np.sin(2 * np.pi * t / periode)      # preference mobile
        T = 23.0 + 3.0 * np.sin(2 * np.pi * t / 37.0)          # balayage
        perceived = (T - p) + rng.normal(0, SIGMA)
        fb = "chaud" if perceived > W else ("froid" if perceived < -W else "ok")
        if k is not None:
            bel = np.convolve(bel, k, mode="same"); bel /= bel.sum()
        bel *= vraisemblance(T, fb); bel /= bel.sum()
        if t >= 300:
            errs.append(abs(np.sum(grid * bel) - p))
    return np.mean(errs)

periodes = [200, 400, 800, 1600, 3200]
drifts = np.array([0.0, 0.02, 0.04, 0.07, 0.10, 0.15, 0.22, 0.30, 0.45, 0.65])

opt_drift, courbes = [], {}
for P in periodes:
    errs = np.array([np.mean([erreur_suivi(P, sd, seed=s) for s in range(3)]) for sd in drifts])
    courbes[P] = errs
    opt_drift.append(drifts[np.argmin(errs)])
    print(f"periode={P:5d} (change lent->rapide)  oubli optimal = {drifts[np.argmin(errs)]:.2f}  "
          f"(erreur min {errs.min():.3f} C)")

# ---------- Figures ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
cols = plt.cm.viridis(np.linspace(0, 0.85, len(periodes)))
for P, c in zip(periodes, cols):
    ax1.plot(drifts, courbes[P], "-o", color=c, ms=4, label=f"periode {P}")
    imin = np.argmin(courbes[P]); ax1.plot(drifts[imin], courbes[P][imin], "o", color=c, ms=11, mfc="none", mew=2)
ax1.set_xlabel("Vitesse d'oubli (sigma_drift)"); ax1.set_ylabel("Erreur de suivi ( C)")
ax1.set_title("Un optimum d'oubli pour chaque volatilite"); ax1.legend(fontsize=8)

volat = 1.0 / np.array(periodes)
ax2.plot(volat, opt_drift, "-o", color="#D85A30", lw=2, ms=7)
ax2.set_xlabel("Volatilite de la preference (1 / periode)")
ax2.set_ylabel("Oubli optimal (sigma_drift)")
ax2.set_title("Plus l'occupant change vite, plus il faut oublier")
for P, v, od in zip(periodes, volat, opt_drift):
    ax2.annotate(f"P={P}", (v, od), textcoords="offset points", xytext=(6, -10), fontsize=8)

plt.tight_layout(); plt.savefig("forgetting_vs_volatility.png", dpi=110)
print("\nFigure enregistree : forgetting_vs_volatility.png")
