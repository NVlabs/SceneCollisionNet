import argparse
import os

import matplotlib
import pandas as pd
from natsort import natsorted

matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Make loss plots for model")
    parser.add_argument(
        "csv_files",
        type=str,
        nargs="+",
        help="path to the csv file with losses",
    )
    parser.add_argument(
        "-o", "--output_path", type=str, help="path to store output images"
    )
    parser.add_argument(
        "-s", "--sample", type=int, default=1, help="downsample plotting rate"
    )
    args = parser.parse_args()

    output_path = (
        os.path.dirname(args.csv_files[0])
        if args.output_path is None
        else args.output_path
    )

    legend = len(args.csv_files) > 1

    f1, axes1 = plt.subplots(2, 2, figsize=(9, 6))
    f2, axes2 = plt.subplots(2, 2, figsize=(9, 6))
    for csvf in natsorted(args.csv_files):

        data = pd.read_csv(csvf)
        data = data[data["iteration"] % args.sample == 0]

        # Make and save training plots
        label = os.path.dirname(csvf).split("/")[-1] if legend else None
        sns.lineplot(
            x="epoch",
            y="train/loss",
            data=data,
            ax=axes1[0, 0],
            label=label,
            legend=False,
        )
        sns.lineplot(
            x="epoch",
            y="train/acc",
            data=data,
            ax=axes1[0, 1],
            label=label,
            legend=False,
        )
        sns.lineplot(
            x="epoch",
            y="train/f1",
            data=data,
            ax=axes1[1, 0],
            label=label,
            legend=False,
        )
        sns.lineplot(
            x="epoch",
            y="train/ap",
            data=data,
            ax=axes1[1, 1],
            label=label,
            legend=False,
        )

        # Make and save validation plots
        sns.lineplot(
            x="epoch",
            y="valid/loss",
            data=data,
            ax=axes2[0, 0],
            label=label,
            legend=False,
        )
        sns.lineplot(
            x="epoch",
            y="valid/acc",
            data=data,
            ax=axes2[0, 1],
            label=label,
            legend=False,
        )
        sns.lineplot(
            x="epoch",
            y="valid/f1",
            data=data,
            ax=axes2[1, 0],
            label=label,
            legend=False,
        )
        sns.lineplot(
            x="epoch",
            y="valid/ap",
            data=data,
            ax=axes2[1, 1],
            label=label,
            legend=False,
        )

    if legend:
        f1.legend(
            *axes1[0, 0].get_legend_handles_labels(),
            loc="center left",
            bbox_to_anchor=(1.0, 0.5)
        )
        f2.legend(
            *axes2[0, 0].get_legend_handles_labels(),
            loc="center left",
            bbox_to_anchor=(1.0, 0.5)
        )

    f1.tight_layout()
    f2.tight_layout()
    f1.savefig(
        os.path.join(output_path, "training_plots.png"), bbox_inches="tight"
    )
    f2.savefig(
        os.path.join(output_path, "validation_plots.png"), bbox_inches="tight"
    )
