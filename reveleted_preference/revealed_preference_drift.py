"""
Semaine 6 (finalisation) : preference REVELEE + qui DERIVE dans le temps
=======================================================================
Un foyer reel : son seuil de confort change avec les saisons, et on ne voit
que son comportement (clim allumee ou non). On combine :
  - la vraisemblance comportementale (semaine 6),
  - l'oubli / filtre bayesien (semaine 5),
pour suivre un seuil de confort MOBILE a partir du seul on/off de la clim.

C'est exactement la situation des donnees de Hanoi (mesures sur ~1 an).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(5)
grid = np.arange(20, 32.001, 0.1); dx = grid[1] - grid[0]
RAIDEUR = 1.2

def noyau(sd):
    d = int(np.ceil(3 * sd / dx)); x = np.arange(-d, d + 1) * dx
    k = np.exp(-0.5 * (x / sd) ** 2); return k / k.sum()

def p_allume(T, seuil):
    return 1.0 / (1.0 + np.exp(-(T - seuil) / RAIDEUR))

class EstimateurRevele:
    """Behavioral likelihood + oubli optionnel (sd>0 => filtre qui suit la derive)."""
    def __init__(self, sigma_drift=0.0):
        self.belief = np.ones_like(grid) / len(grid)
        self.k = noyau(sigma_drift) if sigma_drift > 0 else None
    def observer(self, T, on):
        if self.k is not None:                        # prediction (oubli)
            self.belief = np.convolve(self.belief, self.k, mode="same"); self.belief /= self.belief.sum()
        p = p_allume(T, grid)                          # mise a jour comportementale
        self.belief *= np.clip(p if on else (1 - p), 1e-9, None); self.belief /= self.belief.sum()
    @property
    def estimation(self):
        return float(np.sum(grid * self.belief))

# ---------- Occupant : seuil de confort qui derive (saisonnier) ----------
N = 3000
def seuil_vrai(t):
    return 26.0 + 2.0 * np.sin(2 * np.pi * t / 1500.0)   # derive 24..28 C

naif = EstimateurRevele(sigma_drift=0.0)
filtre = EstimateurRevele(sigma_drift=0.10)
verite, est_naif, est_filtre = [], [], []
for t in range(N):
    s = seuil_vrai(t)
    T = 26.0 + 4.0 * np.sin(2 * np.pi * t / 41.0) + rng.normal(0, 0.4)   # temperature interieure
    on = 1 if rng.random() < p_allume(T, s) else 0
    naif.observer(T, on); filtre.observer(T, on)
    verite.append(s); est_naif.append(naif.estimation); est_filtre.append(filtre.estimation)

verite = np.array(verite); est_naif = np.array(est_naif); est_filtre = np.array(est_filtre)
e_naif = np.mean(np.abs(est_naif[300:] - verite[300:]))
e_filtre = np.mean(np.abs(est_filtre[300:] - verite[300:]))
print(f"Suivi d'un seuil de confort qui derive, a partir du seul comportement clim :")
print(f"   Naif (sans oubli)  : erreur {e_naif:.3f} C")
print(f"   Filtre (avec oubli): erreur {e_filtre:.3f} C")
print(f"   -> {e_naif/e_filtre:.1f}x meilleur")

plt.figure(figsize=(11, 4.4))
plt.plot(verite, "k", lw=2.2, label="Vrai seuil de confort (cache, derive)")
plt.plot(est_naif, color="#378ADD", lw=1.5, label=f"Naif, err={e_naif:.2f}")
plt.plot(est_filtre, color="#D85A30", lw=1.6, label=f"Filtre + oubli, err={e_filtre:.2f}")
plt.xlabel("Pas de temps (mesures)"); plt.ylabel("Seuil de confort ( C)")
plt.title("Suivre un seuil de confort qui derive, a partir du seul on/off de la clim")
plt.legend(loc="lower left", fontsize=9); plt.tight_layout()
plt.savefig("revealed_preference_drift.png", dpi=110)
print("\nFigure enregistree : revealed_preference_drift.png")
