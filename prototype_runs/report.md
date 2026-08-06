# Training-run index

Each important experiment keeps a `report.md` beside its checkpoints and raw
JSON history. Beginning with V4.2, training writes and refreshes that report
automatically after every epoch.

| Experiment | Scope | Status | Main lesson |
| --- | --- | --- | --- |
| `hybrid_v4_bleed` | 45-60 s | Complete | Soft scene-memory correction prevents hard resets |
| `hybrid_v4_1_motion` | 45-60 s | Complete | Dual slow/fast velocity restores more local motion |
| `hybrid_v4_1_full` | Full 219.1 s | Complete | Same architecture collapses episodically on the long timeline |
| `hybrid_v4_2_long_horizon` | Full 219.1 s | Code ready, untrained | Tests physical time, local memory bandwidth, cut-aware bleed, and rollout-state exposure |

Transient smoke-test folders remain useful for debugging but are not treated as
scientific results. The project-wide architecture history and headline numbers
also remain in the root `README.md`.
