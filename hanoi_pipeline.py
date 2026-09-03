"""
Semaine 7 : pipeline "donnees reelles" (Hanoi, 49 appartements)
===============================================================
Chaine complete : charger -> standardiser -> reechantillonner a l'heure
-> estimer le seuil de confort de CHAQUE appartement (preference revelee).

Comme l'archive Mendeley se telecharge a la main, ce script FONCTIONNE des
maintenant sur des donnees FICTIVES imitant la structure de Hanoi. Pour
passer au reel : mets tes CSV dans le dossier DATA_DIR et ajuste COLUMN_MAP.
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

# =====================================================================
# CONFIG a adapter quand tu auras les vrais fichiers
# =====================================================================
DATA_DIR = "hanoi_data"          # dossier contenant les CSV reels (un par appartement)
COLUMN_MAP = {                   # <- a ajuster selon les vrais noms de colonnes
    "timestamp": "timestamp",
    "indoor_temp": "indoor_temperature",
    "ac_status": "ac_status",    # 1 = clim allumee, 0 = eteinte
}

# =====================================================================
# 1) ESTIMATEUR (importe conceptuellement de la semaine 6)
# =====================================================================
grid = np.arange(20, 32.001, 0.1); dx = grid[1] - grid[0]
RAIDEUR = 1.2
def _noyau(sd):
    d = int(np.ceil(3 * sd / dx)); x = np.arange(-d, d + 1) * dx
    k = np.exp(-0.5 * (x / sd) ** 2); return k / k.sum()
def _p_allume(T, seuil):
    return 1.0 / (1.0 + np.exp(-(T - seuil) / RAIDEUR))

def estimer_seuil(temp, ac_on, sigma_drift=0.10):
    """Renvoie la trajectoire du seuil de confort estime pour un appartement."""
    belief = np.ones_like(grid) / len(grid); k = _noyau(sigma_drift)
    traj = []
    for T, on in zip(temp, ac_on):
        if np.isnan(T) or np.isnan(on):
            traj.append(np.sum(grid * belief)); continue
        belief = np.convolve(belief, k, mode="same"); belief /= belief.sum()
        p = _p_allume(T, grid)
        belief *= np.clip(p if on >= 0.5 else (1 - p), 1e-9, None); belief /= belief.sum()
        traj.append(float(np.sum(grid * belief)))
    return np.array(traj)

# =====================================================================
# 2) CHARGEMENT + STANDARDISATION + REECHANTILLONNAGE HORAIRE
# =====================================================================
def charger_appartement(path):
    df = pd.read_csv(path)
    df = df.rename(columns={v: k for k, v in COLUMN_MAP.items()})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    horaire = pd.DataFrame({
        "indoor_temp": df["indoor_temp"].resample("1h").mean(),
        "ac_on": (df["ac_status"].resample("1h").mean() >= 0.5).astype(float),
    })
    return horaire.dropna()

# =====================================================================
# 3) DONNEES FICTIVES (si pas de vrais fichiers) : structure type Hanoi
# =====================================================================
def generer_donnees_fictives(n_appartements=8, jours=120):
    os.makedirs(DATA_DIR, exist_ok=True)
    seuils_vrais = {}
    t = pd.date_range("2020-06-01", periods=jours * 96, freq="15min")  # 15 min
    tt = np.arange(len(t))
    for i in range(n_appartements):
        base = rng.uniform(24.5, 28.0)                         # confort propre au foyer
        seuil = base + 1.5 * np.sin(2 * np.pi * tt / (96 * 90))  # derive saisonniere
        seuils_vrais[f"apt_{i:02d}"] = float(np.mean(seuil))
        t_ext = 30 + 5 * np.sin(2 * np.pi * tt / 96) + 3 * np.sin(2 * np.pi * tt / (96 * 90))
        t_int = 0.5 * t_ext + 13 + 2.5 * np.sin(2 * np.pi * tt / 96) + rng.normal(0, 0.5, len(tt))
        ac = (rng.random(len(tt)) < _p_allume(t_int, seuil)).astype(int)
        pd.DataFrame({
            "timestamp": t, "indoor_temperature": np.round(t_int, 2),
            "outdoor_temperature": np.round(t_ext, 2),
            "relative_humidity": np.round(rng.uniform(60, 90, len(tt)), 1),
            "ac_status": ac,
            "window_status": ((rng.random(len(tt)) < 0.1) & (t_int < seuil - 2)).astype(int),
        }).to_csv(os.path.join(DATA_DIR, f"apt_{i:02d}.csv"), index=False)
    return seuils_vrais

# =====================================================================
# 4) PIPELINE
# =====================================================================
fichiers = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
donnees_reelles = len(fichiers) > 0
if not donnees_reelles:
    print("Aucun fichier reel trouve -> generation de donnees FICTIVES (structure type Hanoi).\n")
    seuils_vrais = generer_donnees_fictives()
    fichiers = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

resultats = {}
for path in fichiers:
    nom = os.path.splitext(os.path.basename(path))[0]
    d = charger_appartement(path)
    traj = estimer_seuil(d["indoor_temp"].values, d["ac_on"].values)
    resultats[nom] = {"traj": traj, "seuil_estime": float(np.mean(traj[-len(traj)//3:])),
                      "n_heures": len(d), "index": d.index}

print(f"{len(resultats)} appartements traites (pas horaire).\n")
print(f"{'Appartement':12} {'Seuil estime':>13}", end="")
if not donnees_reelles: print(f" {'Seuil vrai':>11} {'erreur':>8}")
else: print()
for nom, r in resultats.items():
    line = f"{nom:12} {r['seuil_estime']:>11.2f} C"
    if not donnees_reelles:
        v = seuils_vrais[nom]; line += f" {v:>9.2f} C {abs(r['seuil_estime']-v):>7.2f}"
    print(line)

# =====================================================================
# 5) FIGURES
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

if not donnees_reelles:
    noms = list(resultats.keys())
    est = [resultats[n]["seuil_estime"] for n in noms]
    vrai = [seuils_vrais[n] for n in noms]
    ax1.plot([24, 29], [24, 29], "k--", lw=1, label="ideal (estime = vrai)")
    ax1.scatter(vrai, est, color="#D85A30", s=60, zorder=3)
    ax1.set_xlabel("Seuil de confort VRAI ( C)"); ax1.set_ylabel("Seuil ESTIME ( C)")
    ax1.set_title("Validation : le pipeline retrouve chaque seuil"); ax1.legend(fontsize=9)

premier = list(resultats.values())[0]
ax2.plot(premier["index"], premier["traj"], color="#D85A30", lw=1.4)
ax2.set_xlabel("Temps"); ax2.set_ylabel("Seuil de confort estime ( C)")
ax2.set_title(f"Suivi du seuil dans le temps ({list(resultats.keys())[0]})")
fig.autofmt_xdate()

plt.tight_layout(); plt.savefig("hanoi_pipeline_result.png", dpi=110)
print("\nFigure enregistree : hanoi_pipeline_result.png")
