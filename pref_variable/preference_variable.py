"""
Preference learning avec preference VARIABLE DANS LE TEMPS
=========================================================
Une personne ne prefere pas 25 C en permanence : sa temperature ideale
change (matin/soir, saison...). On compare deux estimateurs sur un occupant
dont la preference se deplace :

  A) NAIF   : l'estimateur bayesien de la semaine 3 (mise a jour seule).
              Il accumule les preuves indefiniment -> devient rigide.
  B) FILTRE : le meme, mais avec une etape d'OUBLI a chaque pas
              (prediction par diffusion), qui lui permet de SUIVRE.

Les deux recoivent EXACTEMENT la meme sequence (temperature, retour),
donc la comparaison est juste.
"""

import numpy as np
from math import erf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)

def Phi(x):
    return 0.5 * (1 + np.vectorize(erf)(x / np.sqrt(2)))

# ---------- Grille des preferences candidates ----------
grid = np.arange(16, 30.001, 0.1)
dx = grid[1] - grid[0]

# ---------- Occupant : parametres ----------
W = 1.0          # demi-largeur de la zone de confort
SIGMA = 1.5      # bruit (incoherence humaine)

def preference_vraie(t, N):
    """La preference CACHEE, qui varie dans le temps.
    Sinusoide lente (cycle jour/nuit) + un saut brutal au milieu."""
    base = 23.0 + 2.0 * np.sin(2 * np.pi * t / 600.0)   # oscille ~21..25
    saut = -2.0 if t > N // 2 else 0.0                  # changement soudain
    return base + saut

def retour_occupant(T, p_true):
    perceived = (T - p_true) + rng.normal(0, SIGMA)
    if perceived > W:  return "hot"
    if perceived < -W: return "cold"
    return "comfortable"

def vraisemblance(T, fb):
    diff = T - grid
    p_hot = 1 - Phi((W - diff) / SIGMA)
    p_cold = Phi((-W - diff) / SIGMA)
    p_ok = np.clip(1 - p_hot - p_cold, 1e-9, None)
    return np.clip({"hot": p_hot, "cold": p_cold, "comfortable": p_ok}[fb], 1e-9, None)

# ---------- Noyau de diffusion (l'oubli) ----------
def noyau_gaussien(sigma_drift):
    """Petit noyau gaussien pour 'flouter' la croyance a chaque pas."""
    demi = int(np.ceil(3 * sigma_drift / dx))
    x = np.arange(-demi, demi + 1) * dx
    k = np.exp(-0.5 * (x / sigma_drift) ** 2)
    return k / k.sum()

# ---------- Un estimateur (naif si sigma_drift=0, filtre sinon) ----------
class Estimateur:
    def __init__(self, sigma_drift=0.0):
        self.belief = np.ones_like(grid) / len(grid)
        self.sigma_drift = sigma_drift
        self.noyau = noyau_gaussien(sigma_drift) if sigma_drift > 0 else None

    def pas(self, T, fb):
        # 1) PREDICTION : oubli (uniquement pour le filtre)
        if self.noyau is not None:
            self.belief = np.convolve(self.belief, self.noyau, mode="same")
            self.belief /= self.belief.sum()
        # 2) MISE A JOUR : le retour resserre la croyance
        self.belief *= vraisemblance(T, fb)
        self.belief /= self.belief.sum()

    @property
    def estimation(self):
        return float(np.sum(grid * self.belief))

# ---------- Experience ----------
N = 2000
naif = Estimateur(sigma_drift=0.0)
filtre = Estimateur(sigma_drift=0.15)

verite, est_naif, est_filtre = [], [], []
for t in range(N):
    p = preference_vraie(t, N)
    T = 23.0 + 3.0 * np.sin(2 * np.pi * t / 37.0)   # controleur : balaie autour de la zone
    fb = retour_occupant(T, p)
    naif.pas(T, fb)
    filtre.pas(T, fb)
    verite.append(p)
    est_naif.append(naif.estimation)
    est_filtre.append(filtre.estimation)

verite = np.array(verite); est_naif = np.array(est_naif); est_filtre = np.array(est_filtre)

# On mesure l'erreur de suivi apres une periode de rodage (200 pas)
err_naif = np.mean(np.abs(est_naif[200:] - verite[200:]))
err_filtre = np.mean(np.abs(est_filtre[200:] - verite[200:]))
print(f"Average tracking error (after warm-up):")
print(f"   NAIVE estimator   : {err_naif:.3f} C")
print(f"   FILTER estimator  : {err_filtre:.3f} C")
print(f"   -> the filter tracks the moving preference {err_naif/err_filtre:.1f}x better")

# ---------- Figure principale ----------
plt.figure(figsize=(11, 4.6))
plt.plot(verite, color="black", lw=2.2, label="True preference (hidden)")
plt.plot(est_naif, color="#378ADD", lw=1.6, alpha=0.9, label=f"Naive (no forgetting), err={err_naif:.2f}")
plt.plot(est_filtre, color="#D85A30", lw=1.6, label=f"Filter (with forgetting), err={err_filtre:.2f}")
plt.axvline(N // 2, color="gray", ls=":", lw=1)
plt.text(N // 2 + 10, 26.2, "sudden jump", color="gray", fontsize=9)
plt.xlabel("Time step"); plt.ylabel("Preferred temperature ( C)")
plt.title("Tracking a changing preference: naive vs filter with forgetting")
plt.legend(loc="lower left", fontsize=9); plt.tight_layout()
plt.savefig("preference_variable.png", dpi=110)
print("\nSaving picture : preference_variable.png")

# ---------- Balayage : effet de la vitesse d'oubli ----------
print("\n  (sigma_drift sweep) :")
for sd in [0.0, 0.03, 0.08, 0.15, 0.30, 0.60]:
    e = Estimateur(sigma_drift=sd)
    errs = []
    r2 = np.random.default_rng(7)
    for t in range(N):
        p = preference_vraie(t, N)
        T = 23.0 + 3.0 * np.sin(2 * np.pi * t / 37.0)
        perceived = (T - p) + r2.normal(0, SIGMA)
        fb = "hot" if perceived > W else ("cold" if perceived < -W else "comfortable")
        e.pas(T, fb)
        if t >= 200:
            errs.append(abs(e.estimation - p))
    tag = "  (= naive)" if sd == 0 else ""
    print(f"   sigma_drift={sd:.2f} -> following error {np.mean(errs):.3f} C{tag}")
