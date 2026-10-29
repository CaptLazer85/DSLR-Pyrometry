
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score


# ==========================================================
# 1. ASK THE USER TO SELECT A CSV FILE
# ==========================================================

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select your calibration CSV file",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)

if not file_path:
    print("No file selected. Program terminated.")
    raise SystemExit

# Read the CSV file
df = pd.read_csv(file_path)


# ==========================================================
# 2. EXTRACT TEMPERATURE AND R/G RATIO
# ==========================================================

temperature_column = "thermocouple_temperature_K"
ratio_column = "red_green_ratio_R_over_G"

# Check whether the required columns exist
required_columns = [temperature_column, ratio_column]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    messagebox.showerror(
        "Missing Columns",
        "The following columns were not found:\n"
        + "\n".join(missing_columns)
    )
    raise SystemExit

# Convert data to numerical values
data = df[[ratio_column, temperature_column]].copy()

data[ratio_column] = pd.to_numeric(
    data[ratio_column], errors="coerce"
)

data[temperature_column] = pd.to_numeric(
    data[temperature_column], errors="coerce"
)

# Remove missing and infinite values
data = data.replace([np.inf, -np.inf], np.nan).dropna()

# Check for sufficient data
if len(data) < 3:
    messagebox.showerror(
        "Insufficient Data",
        "At least 3 valid measurements are required."
    )
    raise SystemExit

x = data[ratio_column].to_numpy(dtype=float)
y = data[temperature_column].to_numpy(dtype=float)

if np.unique(x).size < 2:
    messagebox.showerror(
        "Insufficient Variation",
        "The R/G ratio must contain at least two distinct values."
    )
    raise SystemExit


# ==========================================================
# 3. FIT POLYNOMIALS AND COMPARE THEIR PERFORMANCE
# ==========================================================

# Test degrees 1 to 5, limited by the number of observations.
max_degree = min(5, len(data) - 1)

loo = LeaveOneOut()
results = []

for degree in range(1, max_degree + 1):

    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )

    # Negative MSE is returned, so convert it to RMSE.
    scores = cross_val_score(
        model,
        x.reshape(-1, 1),
        y,
        cv=loo,
        scoring="neg_mean_squared_error"
    )

    loo_rmse = np.sqrt(-scores.mean())

    results.append((degree, loo_rmse))

# Select the degree with the lowest cross-validation RMSE.
best_degree, best_cv_rmse = min(
    results, key=lambda item: item[1]
)


# ==========================================================
# 4. FIT THE SELECTED POLYNOMIAL TO ALL DATA
# ==========================================================

best_model = make_pipeline(
    PolynomialFeatures(
        degree=best_degree,
        include_bias=False
    ),
    LinearRegression()
)

best_model.fit(x.reshape(-1, 1), y)

# Get polynomial coefficients
poly = best_model.named_steps["polynomialfeatures"]
regression = best_model.named_steps["linearregression"]

coefficients = regression.coef_
intercept = regression.intercept_

# Construct the equation
equation = f"T = {intercept:.8g}"

for power, coefficient in enumerate(coefficients, start=1):
    sign = "+" if coefficient >= 0 else "-"
    equation += f" {sign} {abs(coefficient):.8g} x^{power}"

print("\n" + "=" * 65)
print("POLYNOMIAL CALIBRATION: TEMPERATURE vs R/G")
print("=" * 65)

print(f"\nCSV file: {file_path}")
print(f"Number of valid measurements: {len(data)}")
print(f"Selected polynomial degree: {best_degree}")
print(f"LOOCV RMSE: {best_cv_rmse:.4f} K")

print("\nPolynomial equation:")
print(equation)

print("\nCross-validation results:")
for degree, rmse in results:
    print(f"Degree {degree}: LOOCV RMSE = {rmse:.4f} K")

# Calculate R-squared on the complete dataset
y_pred = best_model.predict(x.reshape(-1, 1))

ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)

r_squared = (
    1 - ss_res / ss_tot
    if ss_tot > 0 else float("nan")
)

print(f"\nR-squared on the fitted dataset: {r_squared:.6f}")


# ==========================================================
# 5. GENERATE A SMOOTH FITTED CURVE
# ==========================================================

x_curve = np.linspace(x.min(), x.max(), 500)

y_curve = best_model.predict(
    x_curve.reshape(-1, 1)
)


# ==========================================================
# 6. PLOT EXPERIMENTAL DATA AND FITTED CURVE
# ==========================================================

plt.figure(figsize=(10, 7))

# Experimental data points
plt.scatter(
    x,
    y,
    s=85,
    marker="o",
    edgecolors="black",
    linewidths=0.8,
    label="Experimental data",
    zorder=3
)

# Best fitted polynomial
plt.plot(
    x_curve,
    y_curve,
    linewidth=2.5,
    label=f"Polynomial fit (degree {best_degree})",
    zorder=2
)

plt.xlabel("Red/Green CFA Ratio (R/G)", fontsize=13)
plt.ylabel("Thermocouple Temperature (K)", fontsize=13)

plt.title(
    "Thermocouple Temperature vs. R/G Ratio\n"
    f"Polynomial Degree = {best_degree}, "
    f"R² = {r_squared:.4f}",
    fontsize=14
)

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()

plt.show()


# ==========================================================
# 7. SAVE THE FITTED CURVE AND EQUATION
# ==========================================================

# Save the fitted curve data alongside the original CSV.
output_curve = pd.DataFrame({
    "R_over_G": x_curve,
    "Fitted_Temperature_K": y_curve
})

import os

output_folder = os.path.dirname(file_path)

curve_path = os.path.join(
    output_folder,
    "polynomial_fitted_curve.csv"
)

output_curve.to_csv(curve_path, index=False)

equation_path = os.path.join(
    output_folder,
    "polynomial_equation.txt"
)

with open(equation_path, "w", encoding="utf-8") as file:
    file.write("Temperature vs R/G Polynomial Calibration\n")
    file.write(f"Selected degree: {best_degree}\n")
    file.write(f"LOOCV RMSE: {best_cv_rmse:.6f} K\n")
    file.write(f"R-squared on fitted data: {r_squared:.6f}\n\n")
    file.write(equation + "\n")

print("\nFiles saved successfully:")
print(curve_path)
print(equation_path)

