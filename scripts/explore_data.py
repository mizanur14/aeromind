"""LEGACY Stage-0 exploratory script (full training pool, all 100 engines).

Superseded by ``uv run aeromind eda``. Its outputs in reports/FD001/*.{csv,json,png}
are exploratory only and are not read by any pipeline code.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from aeromind.data.cmapss import load_test_features_only, load_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="data/raw/cmapss",
    )
    parser.add_argument(
        "--dataset",
        default="FD001",
        choices=["FD001", "FD002", "FD003", "FD004"],
    )
    args = parser.parse_args()

    # Legacy exploratory script (full training pool). Test labels are not needed.
    train = load_train(args.data_dir, args.dataset)
    test = load_test_features_only(args.data_dir, args.dataset)

    output = Path("reports") / args.dataset
    output.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": args.dataset,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_engines": int(train.engine_id.nunique()),
        "test_engines": int(test.engine_id.nunique()),
        "train_columns": train.columns.tolist(),
        "train_missing_values": int(train.isna().sum().sum()),
        "train_engine_cycle_min": int(
            train.groupby("engine_id").cycle.max().min()
        ),
        "train_engine_cycle_max": int(
            train.groupby("engine_id").cycle.max().max()
        ),
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(json.dumps(summary, indent=2))

    train.describe().to_csv(
        output / "train_statistics.csv"
    )

    # Plot the run-to-failure RUL trajectory.
    engine = train[train.engine_id == 1]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(engine.cycle, engine.rul)
    ax.set(
        title=f"{args.dataset} — Engine 1 RUL",
        xlabel="Operating cycle",
        ylabel="Remaining useful life (cycles)",
    )
    fig.tight_layout()
    fig.savefig(output / "engine_1_rul.png", dpi=160)
    plt.close(fig)

    # Plot selected sensor trajectories.
    fig, axes = plt.subplots(
        3, 1, figsize=(9, 8), sharex=True
    )

    for ax, sensor in zip(
        axes, ["sensor_2", "sensor_3", "sensor_4"]
    ):
        ax.plot(engine.cycle, engine[sensor])
        ax.set_ylabel(sensor)

    axes[-1].set_xlabel("Operating cycle")
    fig.suptitle(
        f"{args.dataset} — Engine 1 Sensor Trends"
    )
    fig.tight_layout()
    fig.savefig(
        output / "engine_1_sensors.png",
        dpi=160,
    )
    plt.close(fig)

    print(f"\nReports saved to {output.resolve()}")


if __name__ == "__main__":
    main()
