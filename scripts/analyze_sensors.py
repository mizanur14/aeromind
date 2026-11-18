"""LEGACY Stage-0 exploratory script (full training pool, all 100 engines).

Superseded by ``uv run aeromind eda --scope train``. Correlations here include
validation and calibration engines, so they must not drive feature selection.
No pipeline code reads its outputs.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from aeromind.data.cmapss import load_train

train = load_train("data/raw/cmapss", "FD001")

out = Path("reports/FD001")
out.mkdir(parents=True, exist_ok=True)

sensors = [f"sensor_{i}" for i in range(1, 22)]

# Descriptive statistics.
stats = train[sensors].agg(
    ["mean", "std", "min", "max"]
).T

stats["unique_values"] = train[sensors].nunique()

# Standard deviation alone is scale-dependent.
# Count effectively constant channels separately.
stats["effectively_constant"] = (
    (stats["max"] - stats["min"]).abs() < 1e-8
)

stats.to_csv(out / "sensor_statistics.csv")

# Exploratory associations with RUL.
# These are NOT causal explanations.
corr = train[sensors + ["rul"]].corr(
    method="spearman"
)["rul"].drop("rul")

corr = corr.sort_values(key=lambda s: s.abs(), ascending=False)
corr.to_csv(out / "sensor_rul_correlations.csv")

fig, ax = plt.subplots(figsize=(10, 7))
corr.sort_values().plot.barh(ax=ax)
ax.set_xlabel("Spearman correlation with RUL")
ax.set_title("FD001: Exploratory Sensor-RUL Associations")
fig.tight_layout()
fig.savefig(out / "sensor_rul_correlations.png", dpi=160)
plt.close(fig)

# Examine the trajectories of multiple training engines.
selected = [1, 2, 3, 4, 5]
fig, axes = plt.subplots(3, 1, figsize=(11, 9))

for engine_id in selected:
    engine = train[train.engine_id == engine_id]

    for ax, sensor in zip(
        axes, ["sensor_2", "sensor_4", "sensor_11"]
    ):
        ax.plot(
            engine.cycle,
            engine[sensor],
            alpha=0.75,
            label=f"Engine {engine_id}",
        )
        ax.set_ylabel(sensor)

axes[0].legend(ncol=5, fontsize=8)
axes[-1].set_xlabel("Operating cycle")
fig.suptitle("NASA FD001 — Sensor Trajectories")
fig.tight_layout()
fig.savefig(out / "multi_engine_sensors.png", dpi=160)
plt.close(fig)

summary = {
    "sensors": len(sensors),
    "effectively_constant": stats.index[
        stats["effectively_constant"]
    ].tolist(),
    "top_absolute_correlations": {
        str(k): float(v)
        for k, v in corr.head(8).items()
    },
}

(out / "sensor_analysis_summary.json").write_text(
    json.dumps(summary, indent=2)
)

print(json.dumps(summary, indent=2))
print("\nSensor analysis completed.")
