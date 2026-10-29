
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox


# ============================================================
# 1. USER SETTINGS
# ============================================================

# Polynomial calibration equation:
#
# T = 5304.7215 - 8412.3232*x + 4029.5342*x^2
#
# where x = R/G

C0 = 5304.7215
C1 = -8412.3232
C2 = 4029.5342


# Fixed temperature scale for every frame
MIN_TEMPERATURE = 800.0
MAX_TEMPERATURE = 1120.0


# ------------------------------------------------------------
# Intensity thresholds
# ------------------------------------------------------------

RED_THRESHOLD = 40
GREEN_THRESHOLD = 20


# ------------------------------------------------------------
# R/G ratio thresholds
# ------------------------------------------------------------

# These are initial screening limits, not experimentally
# established calibration limits.
MIN_RG_RATIO = 0.80
MAX_RG_RATIO = 1.30


# ------------------------------------------------------------
# Morphological processing
# ------------------------------------------------------------

MORPH_KERNEL_SIZE = 3

OPENING_ITERATIONS = 1
CLOSING_ITERATIONS = 2


# ------------------------------------------------------------
# Contour filtering
# ------------------------------------------------------------

# Minimum accepted contour area is calculated from image size.
# All contours above this limit are retained.
#
# The minimum is deliberately small so that multiple flame
# regions are not discarded simply because they are disconnected.

MIN_COMPONENT_AREA_PIXELS = 8

# Additional image-size-dependent threshold.
MIN_COMPONENT_AREA_FRACTION = 0.000005


# ------------------------------------------------------------
# Display settings
# ------------------------------------------------------------

COLORBAR_WIDTH = 120

SHOW_MASK_WINDOW = True

OUTPUT_CODEC = "mp4v"


# ============================================================
# 2. SELECT INPUT VIDEO
# ============================================================

root = tk.Tk()
root.withdraw()

video_path = filedialog.askopenfilename(
    title="Select Flame Video",
    filetypes=[
        (
            "Video files",
            "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"
        ),
        ("All files", "*.*")
    ]
)

if not video_path:
    messagebox.showinfo(
        "Cancelled",
        "No video was selected."
    )
    root.destroy()
    raise SystemExit


folder = os.path.dirname(video_path)

filename = os.path.splitext(
    os.path.basename(video_path)
)[0]

output_path = os.path.join(
    folder,
    filename + "_temperature_map.mp4"
)


# ============================================================
# 3. OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    messagebox.showerror(
        "Error",
        "Could not open the selected video."
    )
    root.destroy()
    raise SystemExit


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

if fps <= 0:
    fps = 25.0


print("\nVIDEO INFORMATION")
print("-----------------")
print("Input:", video_path)
print("Resolution:", width, "x", height)
print("FPS:", fps)
print("Total frames:", total_frames)

print("\nCALIBRATION EQUATION")
print("--------------------")
print(
    "T = 5304.7215 - 8412.3232*(R/G)"
    " + 4029.5342*(R/G)^2"
)

print("\nFIXED DISPLAY RANGE")
print("-------------------")
print("Minimum:", MIN_TEMPERATURE, "K")
print("Maximum:", MAX_TEMPERATURE, "K")


# ============================================================
# 4. CREATE MORPHOLOGICAL KERNEL
# ============================================================

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
)


# ============================================================
# 5. FLAME MASKING FUNCTION
# ============================================================

def create_flame_mask(frame):
    """
    Create a flame mask using:

    1. Red and green intensity thresholding
    2. Morphological opening
    3. Morphological closing
    4. External contour detection
    5. Minimum contour-area filtering
    6. Filling all accepted contours
    7. Final intensity and R/G ratio filtering

    Unlike a largest-contour-only method, this function
    retains all accepted disconnected flame regions.
    """

    # --------------------------------------------------------
    # A. Extract BGR channels
    # --------------------------------------------------------

    blue = frame[:, :, 0]
    green = frame[:, :, 1]
    red = frame[:, :, 2]


    # --------------------------------------------------------
    # B. Initial intensity thresholding
    # --------------------------------------------------------

    intensity_mask = (
        (red > RED_THRESHOLD) &
        (green > GREEN_THRESHOLD)
    )

    initial_mask = (
        intensity_mask.astype(np.uint8) * 255
    )


    # --------------------------------------------------------
    # C. Morphological opening
    # --------------------------------------------------------

    # Removes small isolated foreground regions and noise.

    opened_mask = cv2.morphologyEx(
        initial_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=OPENING_ITERATIONS
    )


    # --------------------------------------------------------
    # D. Morphological closing
    # --------------------------------------------------------

    # Bridges small gaps and closes small holes in the
    # detected flame regions.

    cleaned_mask = cv2.morphologyEx(
        opened_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=CLOSING_ITERATIONS
    )


    # --------------------------------------------------------
    # E. Find ALL external contours
    # --------------------------------------------------------

    contours, hierarchy = cv2.findContours(
        cleaned_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    # --------------------------------------------------------
    # F. Determine minimum acceptable contour area
    # --------------------------------------------------------

    image_area = frame.shape[0] * frame.shape[1]

    minimum_area = max(
        MIN_COMPONENT_AREA_PIXELS,
        int(
            image_area *
            MIN_COMPONENT_AREA_FRACTION
        )
    )


    # --------------------------------------------------------
    # G. Keep all sufficiently large contours
    # --------------------------------------------------------

    accepted_contours = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area >= minimum_area:
            accepted_contours.append(contour)


    # --------------------------------------------------------
    # H. Fill every accepted contour
    # --------------------------------------------------------

    filled_region_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    if accepted_contours:

        # IMPORTANT:
        # Draw ALL accepted contours.
        # Do not select only the largest contour.

        cv2.drawContours(
            filled_region_mask,
            accepted_contours,
            contourIdx=-1,
            color=255,
            thickness=cv2.FILLED
        )


    # --------------------------------------------------------
    # I. Calculate R/G ratio safely
    # --------------------------------------------------------

    red_float = red.astype(np.float32)
    green_float = green.astype(np.float32)

    rg_ratio = np.zeros(
        (height, width),
        dtype=np.float32
    )

    np.divide(
        red_float,
        green_float,
        out=rg_ratio,
        where=(green_float > 0)
    )


    # --------------------------------------------------------
    # J. Apply R/G ratio filtering
    # --------------------------------------------------------

    ratio_mask = (
        (rg_ratio >= MIN_RG_RATIO) &
        (rg_ratio <= MAX_RG_RATIO)
    )


    # --------------------------------------------------------
    # K. Construct final valid-pixel mask
    # --------------------------------------------------------

    # A pixel must:
    # - Belong to an accepted contour
    # - Pass red/green intensity thresholds
    # - Pass the R/G ratio thresholds

    valid_mask = (
        (filled_region_mask > 0) &
        intensity_mask &
        ratio_mask
    )


    # Return the final mask and the ratio image.
    return valid_mask, rg_ratio, len(accepted_contours)


# ============================================================
# 6. TEMPERATURE CALCULATION FUNCTION
# ============================================================

def calculate_temperature_map(frame):
    """
    Calculate temperature using the polynomial equation
    only at pixels accepted by the flame mask.
    """

    valid_mask, rg_ratio, contour_count = (
        create_flame_mask(frame)
    )

    temperature_map = np.full(
        (height, width),
        np.nan,
        dtype=np.float32
    )

    x = rg_ratio[valid_mask]

    # Apply the calibration equation
    temperature_map[valid_mask] = (
        C0
        + C1 * x
        + C2 * x**2
    )

    return (
        temperature_map,
        valid_mask,
        contour_count
    )


# ============================================================
# 7. CREATE FIXED TEMPERATURE COLOUR BAR
# ============================================================

def create_colorbar(bar_height):
    """
    Create a fixed 800-1120 K colour bar.

    Top    = 1120 K
    Bottom = 800 K
    """

    panel = np.zeros(
        (bar_height, COLORBAR_WIDTH, 3),
        dtype=np.uint8
    )

    # JET: 0 corresponds to blue, 255 to red.
    values = np.linspace(
        255,
        0,
        bar_height,
        dtype=np.uint8
    ).reshape(bar_height, 1)

    grayscale = np.repeat(
        values,
        25,
        axis=1
    )

    colorbar = cv2.applyColorMap(
        grayscale,
        cv2.COLORMAP_JET
    )

    panel[:, 10:35] = colorbar

    cv2.putText(
        panel,
        "Temp (K)",
        (39, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    labels = [
        MAX_TEMPERATURE,
        1040,
        960,
        880,
        MIN_TEMPERATURE
    ]

    for temperature in labels:

        fraction = (
            (temperature - MIN_TEMPERATURE)
            /
            (MAX_TEMPERATURE - MIN_TEMPERATURE)
        )

        y = int(
            (1.0 - fraction) *
            (bar_height - 1)
        )

        y = max(
            10,
            min(bar_height - 5, y)
        )

        cv2.putText(
            panel,
            f"{int(temperature)}",
            (40, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return panel


# ============================================================
# 8. INITIALIZE OUTPUT VIDEO
# ============================================================

output_width = width + COLORBAR_WIDTH

fourcc = cv2.VideoWriter_fourcc(
    *OUTPUT_CODEC
)

writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (output_width, height)
)

if not writer.isOpened():

    cap.release()

    messagebox.showerror(
        "Error",
        "Could not create the output video. "
        "Try a different output codec."
    )

    root.destroy()
    raise SystemExit


# ============================================================
# 9. PROCESS VIDEO FRAME BY FRAME
# ============================================================

frame_number = 0
total_valid_pixels = 0

global_min_temperature = float("inf")
global_max_temperature = float("-inf")

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # ----------------------------------------------------
        # A. Calculate temperatures and sophisticated mask
        # ----------------------------------------------------

        (
            temperature_map,
            valid_mask,
            contour_count
        ) = calculate_temperature_map(frame)


        valid_temperatures = temperature_map[
            valid_mask
        ]

        valid_temperatures = valid_temperatures[
            np.isfinite(valid_temperatures)
        ]


        # ----------------------------------------------------
        # B. Track calculated temperatures
        # ----------------------------------------------------

        if valid_temperatures.size > 0:

            total_valid_pixels += (
                valid_temperatures.size
            )

            global_min_temperature = min(
                global_min_temperature,
                float(
                    np.min(valid_temperatures)
                )
            )

            global_max_temperature = max(
                global_max_temperature,
                float(
                    np.max(valid_temperatures)
                )
            )


        # ----------------------------------------------------
        # C. Normalize using FIXED temperature limits
        # ----------------------------------------------------

        # No frame-by-frame automatic scaling is performed.
        # The same temperature always maps to the same colour.

        clipped_temperature = np.clip(
            np.nan_to_num(
                temperature_map,
                nan=MIN_TEMPERATURE
            ),
            MIN_TEMPERATURE,
            MAX_TEMPERATURE
        )

        normalized = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        normalized[valid_mask] = (
            (
                clipped_temperature[valid_mask]
                - MIN_TEMPERATURE
            )
            /
            (
                MAX_TEMPERATURE
                - MIN_TEMPERATURE
            )
            * 255.0
        ).astype(np.uint8)


        # ----------------------------------------------------
        # D. Apply JET colour map
        # ----------------------------------------------------

        temperature_colored = cv2.applyColorMap(
            normalized,
            cv2.COLORMAP_JET
        )


        # ----------------------------------------------------
        # E. Set all invalid pixels to black
        # ----------------------------------------------------

        temperature_colored[~valid_mask] = (
            0, 0, 0
        )


        # ----------------------------------------------------
        # F. Add frame information
        # ----------------------------------------------------

        cv2.putText(
            temperature_colored,
            f"Frame: {frame_number}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # G. Add colour bar
        # ----------------------------------------------------

        colorbar = create_colorbar(height)

        combined_frame = np.hstack(
            (
                temperature_colored,
                colorbar
            )
        )


        # ----------------------------------------------------
        # H. Save output frame
        # ----------------------------------------------------

        writer.write(combined_frame)


        # ----------------------------------------------------
        # I. Display temperature-map preview
        # ----------------------------------------------------

        preview = cv2.resize(
            combined_frame,
            None,
            fx=0.75,
            fy=0.75
        )

        cv2.imshow(
            "Flame Temperature Map",
            preview
        )


        # ----------------------------------------------------
        # J. Display the segmentation mask
        # ----------------------------------------------------

        if SHOW_MASK_WINDOW:

            mask_display = (
                valid_mask.astype(np.uint8) * 255
            )

            cv2.imshow(
                "Flame Mask - White = Accepted Pixels",
                mask_display
            )


        # Press Q to stop processing
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print(
                "\nProcessing stopped by user."
            )
            break


        if frame_number % 100 == 0:

            print(
                f"Processed {frame_number}"
                f" / {total_frames} frames"
            )


finally:

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    root.destroy()


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\nPROCESSING SUMMARY")
print("------------------")

print("Output video:", output_path)

print("Frames processed:", frame_number)

print(
    "Total accepted pixel observations:",
    total_valid_pixels
)

print(
    "Fixed display range:",
    f"{MIN_TEMPERATURE:.0f}–"
    f"{MAX_TEMPERATURE:.0f} K"
)

if total_valid_pixels > 0:

    print(
        "Minimum calculated temperature before "
        "display clipping:",
        f"{global_min_temperature:.2f} K"
    )

    print(
        "Maximum calculated temperature before "
        "display clipping:",
        f"{global_max_temperature:.2f} K"
    )

else:

    print(
        "WARNING: No pixels passed the mask. "
        "Check the thresholds and R/G ratio limits."
    )

print("\nProcessing finished.")

