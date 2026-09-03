"""
Pipeline REEL corrige : le seuil de confort d'un foyer est la temperature
a laquelle il COMMENCE a allumer la clim (front de montee de P(clim ON | T)),
et non le point median d'une logistique (biaise car la clim refroidit la piece).
"""
import glob, os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Detection automatique du dossier de donnees ---
# Place ce script dans le meme dossier que tes donnees (peu importe le nom des
# sous-dossiers). Il cherche tout seul les dossiers Dataset_AP1, Dataset_AP2, ...
HERE = os.path.dirname(os.path.abspath(__file__))
LEVEL = 0.25   # on definit le "declenchement" comme le passage de P(clim ON) au-dessus de 25%

def load(path):
    df = pd.read_csv(path, sep=";", usecols=["T_Indoor","AC_State","Occupancy_State"])
    d = df[df.Occupancy_State == 1]
    return d["T_Indoor"].values, d["AC_State"].values

def pon_curve(T, on, tmin=22, tmax=35, bw=1.0, min_n=40):
    bins = np.arange(tmin, tmax + bw, bw); centers = bins[:-1] + bw/2
    p = np.full(len(centers), np.nan)
    for i in range(len(centers)):
        m = (T >= bins[i]) & (T < bins[i+1])
        if m.sum() >= min_n: p[i] = on[m].mean()
    return centers, p

def onset_threshold(centers, p, level=LEVEL):
    """Front de montee : 1re temperature ou P(clim ON) franchit `level` en montant."""
    for i in range(1, len(centers)):
        if np.isnan(p[i-1]) or np.isnan(p[i]): continue
        if p[i-1] < level <= p[i]:
            return centers[i-1] + (level - p[i-1]) / (p[i] - p[i-1]) * (centers[i] - centers[i-1])
    return np.nan

# ---------- Traitement ----------
files = sorted(glob.glob(os.path.join(HERE, "**", "Dataset_AP*", "*.csv"), recursive=True))
if not files:
    raise SystemExit("No files found. Check that the Dataset_AP* folders "
                     "are indeed located somewhere under: " + HERE)
print("CSV files found:", len(files))
thresholds, examples = {}, {}
for f in files:
    name = os.path.splitext(os.path.basename(f))[0]
    T, on = load(f)
    if len(T) < 500 or on.sum() < 50: continue
    frac = on.mean()
    if not (0.02 < frac < 0.98): continue
    c, p = pon_curve(T, on)
    s = onset_threshold(c, p)
    if np.isnan(s): continue
    thresholds[name] = s
    examples[name] = (c, p, frac)

vals = np.array(list(thresholds.values()))
print(f"Households with an estimable comfort threshold: {len(vals)}")
print(f"AC onset temperature (= revealed comfort threshold):")
print(f"   mean {vals.mean():.1f} C | median {np.median(vals):.1f} C | "
      f"min {vals.min():.1f} | max {vals.max():.1f} | std dev {vals.std():.1f}")
print(f"\nComparison: a classic PMV setpoint is ~24 C.")
print(f"-> Hanoi households tolerate on average {vals.mean()-24:.1f} C more (adaptive comfort).")
order = sorted(thresholds.items(), key=lambda kv: kv[1])
print("\nMost 'cold-sensitive' households:", [f'{n}={s:.1f}' for n,s in order[:3]])
print("Most tolerant households:", [f'{n}={s:.1f}' for n,s in order[-3:]])

# ---------- Figures ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.7))

ax1.hist(vals, bins=np.arange(24, 31, 0.5), color="#D85A30", edgecolor="white")
ax1.axvline(vals.mean(), color="black", ls="--", lw=2, label=f"mean {vals.mean():.1f} C")
ax1.axvline(24.0, color="#378ADD", ls=":", lw=2, label="typical PMV setpoint ~24 C")
ax1.set_xlabel("Revealed comfort threshold ( C)"); ax1.set_ylabel("Number of households")
ax1.set_title(f"Diversity of real preferences ({len(vals)} Hanoi households)")
ax1.legend(fontsize=9)

for name in ["AP11_BR", "AP17_BR", "AP26_BR"]:
    if name in examples:
        c, p, frac = examples[name]
        ax2.plot(c, p, "-o", ms=4, label=f"{name} (threshold {thresholds[name]:.1f} C)")
ax2.axhline(LEVEL, color="gray", ls=":", label=f"onset level ({LEVEL})")
ax2.set_xlabel("Indoor temperature ( C)"); ax2.set_ylabel("P(AC on | occupant present)")
ax2.set_title("Real behavior is bell-shaped (the AC cools the room)")
ax2.legend(fontsize=8)

plt.tight_layout(); plt.savefig("hanoi_real_onset.png", dpi=110)
print("\nFigure saved: hanoi_real_onset.png")
