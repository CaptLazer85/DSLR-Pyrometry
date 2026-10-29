import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneOut

from tkinter import Tk, filedialog
import os


# --------------------------------------------------
# 1. Select the CSV file
# --------------------------------------------------

root = Tk()
root.withdraw()

csv_path = filedialog.askopenfilename(
    title="Select your CFA calibration CSV file",
    filetypes=[("CSV files", "*.csv")]
)

root.destroy()

if not csv_path:
    raise SystemExit("No CSV file selected.")


# --------------------------------------------------
# 2. Read the experimental data
# --------------------------------------------------

df = pd.read_csv(csv_path)

x_column = "red_green_ratio_R_over_G"
y_column = "thermocouple_temperature_K"

if x_column not in df.columns or y_column not in df.columns:
    raise ValueError(
        "The CSV must contain the columns:\n"
        f"{x_column}\n{y_column}"
    )

data = df[[x_column, y_column]].copy()

data[x_column] = pd.to_numeric(
    data[x_column], errors="coerce"
)

data[y_column] = pd.to_numeric(
    data[y_column], errors="coerce"
)

data = data.replace([np.inf, -np.inf], np.nan).dropna()

x = data[x_column].to_numpy(dtype=float)
y = data[y_column].to_numpy(dtype=float)

if len(x) < 3:
    raise ValueError("At least three valid observations are required.")


# --------------------------------------------------
# 3. Define the exponential model
# --------------------------------------------------

def exponential_model(x, a, b):
    return a * np.exp(b * x)


# --------------------------------------------------
# 4. Fit the exponential curve
# --------------------------------------------------

initial_guess = [600, 0.4]

parameters, covariance = curve_fit(
    exponential_model,
    x,
    y,
    p0=initial_guess,
    maxfev=100000
)

a, b = parameters

y_pred = exponential_model(x, a, b)


# --------------------------------------------------
# 5. Calculate fitting performance
# --------------------------------------------------

r2 = r2_score(y, y_pred)

rmse = np.sqrt(mean_squared_error(y, y_pred))

mae = mean_absolute_error(y, y_pred)


# --------------------------------------------------
# 6. Leave-one-out cross-validation
# --------------------------------------------------

loo = LeaveOneOut()
loo_predictions = []

for train_indices, test_indices in loo.split(x):

    x_train = x[train_indices]
    y_train = y[train_indices]

    x_test = x[test_indices]

    try:
        fold_parameters, _ = curve_fit(
            exponential_model,
            x_train,
            y_train,
            p0=parameters,
            maxfev=100000
        )

        prediction = exponential_model(
            x_test, *fold_parameters
        )[0]

        loo_predictions.append(prediction)

    except (RuntimeError, ValueError):
        loo_predictions.append(np.nan)

loo_predictions = np.array(loo_predictions)

valid = np.isfinite(loo_predictions)

if valid.all():
    loo_rmse = np.sqrt(
        mean_squared_error(y, loo_predictions)
    )
else:
    loo_rmse = np.nan


# --------------------------------------------------
# 7. Print the results
# --------------------------------------------------

print("\n========== EXPONENTIAL CURVE FITTING ==========")

print(f"Number of observations: {len(x)}")

print("\nFitted equation:")

print(f"T = {a:.6f} * exp({b:.6f} * (R/G))")

print("\nPerformance metrics:")

print(f"R-squared (R²): {r2:.4f}")
print(f"RMSE: {rmse:.4f} K")
print(f"MAE: {mae:.4f} K")

if np.isfinite(loo_rmse):
    print(f"LOOCV RMSE: {loo_rmse:.4f} K")
else:
    print("LOOCV RMSE could not be calculated reliably.")

print("\n================================================")


# --------------------------------------------------
# 8. Generate the fitted curve
# --------------------------------------------------

x_curve = np.linspace(x.min(), x.max(), 300)

y_curve = exponential_model(x_curve, a, b)


# --------------------------------------------------
# 9. Plot experimental data and fitted curve
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.scatter(
    x, y,
    label="Experimental data",
    marker="o"
)

plt.plot(
    x_curve, y_curve,
    label="Exponential fit",
    linewidth=2
)

equation_text = (
    f"T = {a:.3f} exp({b:.4f} x)\n"
    f"$R^2$ = {r2:.3f}\n"
    f"RMSE = {rmse:.2f} K"
)

plt.text(
    0.04, 0.95,
    equation_text,
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", alpha=0.15)
)

plt.xlabel("Red-to-green CFA ratio (R/G)")
plt.ylabel("Thermocouple temperature (K)")
plt.title("Exponential Calibration Curve for CFA Pyrometry")

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()


# --------------------------------------------------
# 10. Save the plot and fitted curve
# --------------------------------------------------

folder = os.path.dirname(csv_path)

plot_path = os.path.join(
    folder, "exponential_calibration_curve.png"
)

curve_path = os.path.join(
    folder, "exponential_fitted_curve.csv"
)

plt.savefig(plot_path, dpi=300)

curve_df = pd.DataFrame({
    "R_over_G": x_curve,
    "Fitted_temperature_K": y_curve
})

curve_df.to_csv(curve_path, index=False)

plt.show()

print("\nFiles saved:")
print(plot_path)
print(curve_path)