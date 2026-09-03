"""
Step 1: Small thermal simulator for a room
------------------------------------------------------------
On modélise UNE pièce avec le modèle "RC"
Idée : la température intérieure évolue selon
  - les fuites vers l'extérieur (isolation R),
  - la chaleur du HVAC (chauffage / clim),
  - les apports gratuits (occupant + soleil).

Pour l'instant on pilote la pièce avec un thermostat classique.
Plus tard, on remplacera ce thermostat par un agent RL.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Building Setting (RC model) ---
R  = 5.0    # termal resistor (°C/kW) : plus c'est grand, mieux c'est isolé
C  = 0.5    # termal capacity (kWh/°C) : l'inertie, la "lourdeur" thermique
dt = 0.25   # Time step (h) = 15 min

# --- Time : 2 days ---
n_steps = int(48 / dt)
heures  = np.arange(n_steps) * dt

# --- External temperature : cold on the night, warm in the afternoon ---
T_ext = 8 + 6 * np.sin(2 * np.pi * (heures - 9) / 24)   # oscille entre ~2°C et ~14°C

# --- Target comfort ---
T_consigne = 21.0   # température souhaitée
zone_morte = 0.5    # on n'agit que si on s'écarte de +/- 0.5°C (évite de sur-réagir)


def thermostat(T_in):
    """Contrôleur simple 'tout ou rien' : notre référence à battre plus tard."""
    if T_in < T_consigne - zone_morte:
        return +3.0   # chauffage à fond (kW)
    elif T_in > T_consigne + zone_morte:
        return -3.0   # climatisation à fond (kW)
    else:
        return 0.0    # on ne fait rien


# --- Boucle de simulation (c'est le coeur : la même boucle que le schéma) ---
T_in = 15.0          # température intérieure de départ (pièce froide)
hist_T, hist_Q = [], []
energie_totale = 0.0

for t in range(n_steps):
    Q_hvac  = thermostat(T_in)                                    # 1) le contrôleur décide
    Q_occ   = 0.1                                                 # apport occupant (kW)
    Q_solar = max(0.0, 0.5 * np.sin(2 * np.pi * (heures[t] - 6) / 24))  # soleil en journée

    # 2) l'équation RC : de combien la température change en un pas de temps
    dT = (dt / C) * ((T_ext[t] - T_in) / R + Q_hvac + Q_occ + Q_solar)
    T_in += dT                                                    # 3) on met à jour la pièce

    energie_totale += abs(Q_hvac) * dt   # énergie consommée (kWh) : ce que le RL cherchera à réduire
    hist_T.append(T_in)
    hist_Q.append(Q_hvac)

print(f"Énergie HVAC consommée sur 2 jours : {energie_totale:.1f} kWh")

# --- Display ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax1.plot(heures, hist_T, label="Température intérieure", color="#D85A30", linewidth=2)
ax1.plot(heures, T_ext, label="Température extérieure", color="#378ADD", linestyle="--")
ax1.axhline(T_consigne, color="gray", linestyle=":", label="Consigne (21°C)")
ax1.set_ylabel("Température (°C)")
ax1.legend(loc="upper right")
ax1.set_title("Le simulateur en action : la pièce suit-elle la consigne ?")

ax2.step(heures, hist_Q, where="post", color="#1D9E75", linewidth=1.5)
ax2.set_ylabel("Action HVAC (kW)")
ax2.set_xlabel("Temps (heures)")
ax2.axhline(0, color="gray", linewidth=0.5)
ax2.set_title("Ce que fait le HVAC (+ = chauffe, - = refroidit)")

plt.tight_layout()
plt.savefig("resultat_simulation.png", dpi=100)
plt.show()