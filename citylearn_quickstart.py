"""
Semaine 1 — Prise en main de CityLearn.
Adapté du QuickStart officiel : https://www.citylearn.net/quickstart.html

Objectif : faire tourner UN épisode complet avec le contrôleur de référence
(BaselineAgent, sans RL) et regarder la structure de l'environnement + les KPIs.
"""

from citylearn.citylearn import CityLearnEnv
from citylearn.agents.base import BaselineAgent as Agent

# --- 1) Quels jeux de données sont fournis avec CityLearn ? ---
from citylearn.data import DataSet
try:
    names = DataSet.get_dataset_names()
except TypeError:
    names = DataSet().get_dataset_names()
print("Quelques datasets disponibles :")
for n in names[:12]:
    print("   -", n)
print("   ... et notamment ceux avec 'set_points' (pertinents pour le confort).\n")

# --- 2) Créer l'environnement ---
# central_agent=True : un seul agent contrôle tout le quartier (plus simple pour débuter).
dataset_name = "citylearn_challenge_2023_phase_1"
env = CityLearnEnv(dataset_name, central_agent=True)

# --- 3) Regarder la STRUCTURE (le vrai but de la semaine 1) ---
print(f"Nombre de bâtiments : {len(env.buildings)}")
print(f"Espace d'observation : {env.observation_space[0].shape}")
print(f"Espace d'action      : {env.action_space[0].shape}")
try:
    print("\nNoms des observations (ce que l'agent 'voit') :")
    print(env.observation_names[0])
    print("\nNoms des actions (ce que l'agent 'contrôle') :")
    print(env.action_names[0])
except Exception as e:
    print("(noms d'observations/actions non disponibles dans cette version)", e)

# --- 4) Faire tourner un épisode complet avec le contrôleur de référence ---
model = Agent(env)
observations, _ = env.reset()
while not env.terminated:
    actions = model.predict(observations)
    observations, reward, terminated, truncated, info = env.step(actions)

# --- 5) Afficher les indicateurs de performance (KPIs) ---
kpis = env.evaluate()
kpis = kpis.pivot(index="cost_function", columns="name", values="value").round(3)
kpis = kpis.dropna(how="all")
print("\n===== KPIs (référence Baseline) =====")
print(kpis)
