
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------

file_path = r"thermocouple_raw_cfa_dataset.csv"

df = pd.read_csv(file_path)

# --------------------------------------------------
# 2. Define the column names
# --------------------------------------------------

temperature_column = "thermocouple_temperature_K"

ratio_columns = {
    "red_green_ratio_R_over_G": "Red/Green Ratio (R/G)",
    "blue_red_ratio_B_over_R": "Blue/Red Ratio (B/R)",
    "green_blue_ratio_G_over_B": "Green/Blue Ratio (G/B)"
}

# --------------------------------------------------
# 3. Generate three separate scatter plots
# --------------------------------------------------

for ratio_column, ratio_label in ratio_columns.items():

    # Select the required columns
    plot_data = df[
        [temperature_column, ratio_column]
    ].copy()

    # Convert the columns to numeric values
    plot_data[temperature_column] = pd.to_numeric(
        plot_data[temperature_column], errors="coerce"
    )

    plot_data[ratio_column] = pd.to_numeric(
        plot_data[ratio_column], errors="coerce"
    )

    # Remove missing and infinite values
    plot_data = plot_data.dropna()

    plot_data = plot_data[
        plot_data[temperature_column].map(
            lambda x: abs(x) != float("inf")
        )
        &
        plot_data[ratio_column].map(
            lambda x: abs(x) != float("inf")
        )
    ]

    if plot_data.empty:
        print(f"No valid data available for {ratio_label}.")
        continue

    # Create a separate figure for each ratio
    plt.figure(figsize=(9, 6))

    # Scatter plot: clearly visible measurement points
    plt.scatter(
        plot_data[ratio_column],
        plot_data[temperature_column],
        s=90,                  # Marker size
        marker="o",
        edgecolors="black",
        linewidths=0.8,
        alpha=0.9,
        label="Measurements"
    )

    # Titles and axis labels
    plt.title(
        f"Thermocouple Temperature vs. {ratio_label}",
        fontsize=14
    )

    plt.xlabel(ratio_label, fontsize=12)

    plt.ylabel(
        "Thermocouple Temperature (K)",
        fontsize=12
    )

    # Grid and formatting
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tick_params(axis="both", labelsize=11)
    plt.legend()
    plt.tight_layout()

    # Display the graph
    plt.show()
