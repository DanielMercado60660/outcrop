"""Render a synthetic equivalent-source prediction using the Metal implementation."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_bench import PROJECT, call, make_inputs


def main() -> None:
    side = 120
    args = make_inputs(observations=1200, sources=100, evaluation=side * side)
    prediction = call("metal", "equivalent_sources", **args)["prediction"].reshape(side, side)
    coordinates = args["coordinates"]
    fig, ax = plt.subplots(figsize=(8.4, 6.2), layout="constrained")
    image = ax.imshow(prediction, origin="lower", extent=(0, 1000, 0, 800),
                      cmap="viridis", aspect="equal", interpolation="bilinear")
    ax.scatter(coordinates[::8, 0], coordinates[::8, 1], s=8, facecolors="none",
               edgecolors="white", linewidths=0.45, alpha=0.65, label="Survey samples")
    ax.set(xlabel="Easting (m)", ylabel="Northing (m)",
           title="Equivalent sources · predicted field at 120 m")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.86, pad=0.025)
    colorbar.set_label("Field (arbitrary units)")
    fig.savefig(PROJECT / "figure.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
