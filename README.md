
# Unitree Go2 Foot Contact Probability

Implementation and testing archive for the Go2 Foot Contact Probability Algorithm using MuJoCo.

```text
mujoco_menagerie/
├── main.py             : Main entry (contact estimation + balance controller, keyboard teleop) — WIP for gait
├── demo_balance.py     : QP balance push-recovery demo (records results/qp_balance_test.mp4)
├── claude_sim.py       : Go2Sim wrapper, KeyboardController, contact helpers (debugged with Claude Opus 4.5)
├── algorithm_draft.py  : [Archive] Algorithm-only implementation (first draft, no control/viewer)
├── main_templete.py    : [Archive] Main template
├── graph.py            : Plotting utilities
├── generate_gallery.py : [Upstream] mujoco_menagerie model gallery generator
├── tutorial.ipynb      : MuJoCo tutorial notebook
├── core/               : Algorithm implementation (contact probability, balance control, QP)
│   ├── control.py              : JointController (PD), GaitController (balance / swing)
│   ├── qp.py                   : GRF QP solver (OSQP)
│   ├── foot_mechanics_claude.py: Foot force / height / Jacobians
│   ├── foot_mechanics.py       : [Archive] Previous foot mechanics
│   ├── ground_contact.py       : Contact probability model
│   ├── fsm.py                  : Quadruped contact FSM
│   └── util.py                 : Rotation utilities (Euler, quaternion, SO(3) log)
├── dbug/               : Debugging scripts
├── prototypes/         : Unit tests and component experiments
├── unitree_go2/        : GO2 robot model files
├── models/             : Additional model files
├── results/            : Generated result plots and videos
└── README.md
```
