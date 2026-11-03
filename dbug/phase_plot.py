# debug only: plot ContactModel phase (cm.phi / cm.s_phi / prior) over main.py's SIMUL_TIME
import sys, types
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from core.ground_contact import ContactModel

cm = ContactModel()
dt, T_END = 0.002, 5.0          # main.py: timestep 0.002, SIMUL_TIME 5.0
t = np.arange(0, T_END, dt)
phi, s_phi, prior = [], [], []
for ti in t:
    prior.append(cm.prediction_prob_model(types.SimpleNamespace(time=ti)).ravel())
    phi.append(cm.phi.copy()); s_phi.append(cm.s_phi.copy())
phi, s_phi, prior = map(np.array, (phi, s_phi, prior))

names = ["FL", "FR", "RL", "RR"]
fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
for i in range(4):
    ax[0].plot(t, phi[:, i], label=names[i], ls="--" if i in (1, 2) else "-")
    ax[2].plot(t, prior[:, i], label=names[i], ls="--" if i in (1, 2) else "-")
    ax[1].fill_between(t, i, i + 0.8 * s_phi[:, i], step="post", alpha=0.7)
ax[0].axhline(cm.THRESHOLD, color="k", lw=0.8, ls=":"); ax[0].text(0.02, cm.THRESHOLD + 0.02, f"THRESHOLD={cm.THRESHOLD}")
ax[0].set_ylabel("cm.phi"); ax[0].legend(ncol=4, loc="lower right")
ax[1].set_yticks(np.arange(4) + 0.4, names); ax[1].set_ylabel("cm.s_phi\n(filled = stance)")
ax[2].set_ylabel("prior u\n(prediction_prob_model)"); ax[2].set_xlabel("sim time [s]"); ax[2].legend(ncol=4, loc="lower right")
fig.suptitle(f"ContactModel phase: T={cm.T}s, offset 0.5 (FL/RR vs FR/RL), stance if phi<{cm.THRESHOLD}")
fig.tight_layout(); out = sys.argv[1] if len(sys.argv) > 1 else "results/phase_plot.png"; fig.savefig(out, dpi=110); print("saved:", out)
print("stance ratio per leg:", s_phi.mean(0))
