import os
import csv
import cv2
import rawpy
import numpy as np
import tkinter as tk

from tkinter import filedialog


# ============================================================
# USER SETTINGS
# ============================================================

OUTPUT_CSV = "thermocouple_raw_cfa_dataset.csv"

# Radius around each click, measured in ORIGINAL SENSOR PIXELS.
# Radius 4 means a 9 x 9 neighbourhood around each click.
NEIGHBORHOOD_RADIUS = 2

# Flag the thermocouple location as saturated if ANY RAW sample
# in its selected neighbourhood reaches this fraction of the
# black-corrected sensor range.
#
# 0.99 = conservative warning before the reported white level.
# If flagged, the measurement is NOT added as an accepted calibration point;
# a REJECTED_SATURATION row is still logged in the CSV.
SATURATION_FRACTION = 0.99

WINDOW_NAME = "RAW CFA Mouse Pixel Selection"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
INITIAL_ZOOM = 1.0


# ============================================================
# SELECT CR2 IMAGES
# ============================================================

root = tk.Tk()
root.withdraw()

file_paths = filedialog.askopenfilenames(
    title="Select Canon CR2 flame images",
    filetypes=[
        ("Canon RAW images", "*.CR2 *.cr2"),
        ("All files", "*.*")
    ]
)

root.destroy()

if not file_paths:
    print("No images selected.")
    raise SystemExit

file_paths = sorted(file_paths)
print(f"\nSelected {len(file_paths)} images.")


# ============================================================
# LOAD RAW CFA MOSAIC
# ============================================================

def load_raw_image(file_path):
    with rawpy.imread(file_path) as raw:
        raw_data = raw.raw_image_visible.copy()
        color_indices = raw.raw_colors_visible.copy()

        color_desc = raw.color_desc
        if isinstance(color_desc, bytes):
            color_desc = color_desc.decode("ascii")

        raw_pattern = raw.raw_pattern.copy()
        black_levels = np.asarray(
            raw.black_level_per_channel, dtype=np.float32
        )
        white_level = float(raw.white_level)

        # Display preview only; all measurements use raw_data.
        preview_rgb = raw.postprocess(
            user_flip=0,
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=8
        )

    if raw_data.shape != color_indices.shape:
        raise ValueError("RAW mosaic and CFA index dimensions do not match.")
    if raw_data.shape != preview_rgb.shape[:2]:
        raise ValueError(
            "RAW and preview dimensions do not match. "
            "Check orientation/alignment."
        )
    if len(black_levels) <= int(np.max(color_indices)):
        raise ValueError("Black-level metadata does not cover all CFA indices.")

    red_indices = [
        i for i, c in enumerate(color_desc) if c.upper() == "R"
    ]
    green_indices = [
        i for i, c in enumerate(color_desc) if c.upper() == "G"
    ]
    blue_indices = [
        i for i, c in enumerate(color_desc) if c.upper() == "B"
    ]
    if not red_indices or not green_indices or not blue_indices:
        raise ValueError(
            f"Could not identify all R/G/B CFA channels: {color_desc}"
        )

    raw_float = raw_data.astype(np.float32)
    black_map = black_levels[color_indices]
    corrected_signal = np.maximum(raw_float - black_map, 0.0)

    # Per-pixel threshold in the original RAW-value domain.
    saturation_threshold = (
        black_map + SATURATION_FRACTION * (white_level - black_map)
    )

    # This mask is used to reject near-saturated samples from calculations.
    not_saturated = raw_float < saturation_threshold

    red_mask = np.isin(color_indices, red_indices)
    green_mask = np.isin(color_indices, green_indices)
    blue_mask = np.isin(color_indices, blue_indices)

    print("\nFile:", os.path.basename(file_path))
    print("CFA description:", color_desc)
    print("CFA pattern:")
    for row in raw_pattern:
        print(" ".join(color_desc[int(i)] for i in row))
    print("Black levels:", black_levels)
    print("White level:", white_level)
    print("Red CFA indices:", red_indices)
    print("Green CFA indices:", green_indices)
    print("Blue CFA indices:", blue_indices)

    return {
        "raw_data": raw_data,
        "color_indices": color_indices,
        "corrected_signal": corrected_signal,
        "not_saturated": not_saturated,
        "saturation_threshold": saturation_threshold,
        "red_mask": red_mask,
        "green_mask": green_mask,
        "blue_mask": blue_mask,
        "preview_rgb": preview_rgb,
        "white_level": white_level,
    }


# ============================================================
# INTERACTIVE MOUSE SELECTOR
# ============================================================

class MouseSelector:
    def __init__(self, preview_rgb):
        self.preview_rgb = preview_rgb
        self.height, self.width = preview_rgb.shape[:2]

        self.base_scale = min(
            WINDOW_WIDTH / self.width,
            WINDOW_HEIGHT / self.height
        )
        self.zoom = INITIAL_ZOOM
        self.center_x = self.width / 2
        self.center_y = self.height / 2
        self.points = []

        self.right_dragging = False
        self.last_mouse_position = None

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

    def screen_to_image(self, sx, sy):
        scale = self.base_scale * self.zoom
        x = self.center_x + (sx - WINDOW_WIDTH / 2) / scale
        y = self.center_y + (sy - WINDOW_HEIGHT / 2) / scale
        x = int(round(x))
        y = int(round(y))
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        return x, y

    def image_to_screen(self, x, y):
        scale = self.base_scale * self.zoom
        sx = int(WINDOW_WIDTH / 2 + (x - self.center_x) * scale)
        sy = int(WINDOW_HEIGHT / 2 + (y - self.center_y) * scale)
        return sx, sy

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ix, iy = self.screen_to_image(x, y)
            self.points.append((ix, iy))
            print(f"Selected point {len(self.points)}: x={ix}, y={iy}")

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.right_dragging = True
            self.last_mouse_position = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.right_dragging:
            if self.last_mouse_position is not None:
                old_x, old_y = self.last_mouse_position
                scale = self.base_scale * self.zoom
                self.center_x -= (x - old_x) / scale
                self.center_y -= (y - old_y) / scale
            self.last_mouse_position = (x, y)

        elif event == cv2.EVENT_RBUTTONUP:
            self.right_dragging = False
            self.last_mouse_position = None

        elif event == cv2.EVENT_MOUSEWHEEL:
            old_zoom = self.zoom
            ix_before, iy_before = self.screen_to_image(x, y)

            if flags > 0:
                self.zoom *= 1.25
            else:
                self.zoom /= 1.25

            self.zoom = max(0.5, min(self.zoom, 20.0))

            # Keep the exact original-image coordinate under the cursor fixed.
            # Do not clamp the centre here: clamping would make zooming near an
            # edge jump. The renderer safely displays black outside the image.
            new_scale = self.base_scale * self.zoom
            self.center_x = ix_before - (x - WINDOW_WIDTH / 2) / new_scale
            self.center_y = iy_before - (y - WINDOW_HEIGHT / 2) / new_scale

    def render(self):
        """Render using one consistent image<->screen coordinate transform.

        The old implementation cropped the image and resized that crop to the
        entire window. When the crop touched an image edge, the crop became
        smaller but was still stretched to the full window, while mouse and
        marker coordinates continued using the original scale. This mismatch
        caused selected markers to appear to shift during zooming.

        warpAffine keeps the scale and translation identical for rendering,
        mouse clicks, and marker positions. Areas outside the image are black.
        """
        scale = self.base_scale * self.zoom
        matrix = np.array(
            [
                [scale, 0.0, WINDOW_WIDTH / 2.0 - self.center_x * scale],
                [0.0, scale, WINDOW_HEIGHT / 2.0 - self.center_y * scale],
            ],
            dtype=np.float32,
        )

        preview_bgr = cv2.cvtColor(self.preview_rgb, cv2.COLOR_RGB2BGR)
        display = cv2.warpAffine(
            preview_bgr,
            matrix,
            (WINDOW_WIDTH, WINDOW_HEIGHT),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # Draw each selected point using the exact inverse transform used by
        # screen_to_image(), so the marker stays attached to the same sensor
        # coordinate when zooming or panning.
        for i, (ix, iy) in enumerate(self.points, start=1):
            sx, sy = self.image_to_screen(ix, iy)
            if 0 <= sx < WINDOW_WIDTH and 0 <= sy < WINDOW_HEIGHT:
                cv2.circle(display, (sx, sy), 5, (0, 0, 255), -1)
                cv2.putText(
                    display, str(i), (sx + 6, sy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                )

        instructions = [
            "LEFT CLICK: Select flame points",
            "MOUSE WHEEL: Zoom in/out",
            "RIGHT-DRAG: Pan image",
            "T: Finish this thermocouple location",
            "N: Finish image",
            "Q / ESC: Quit"
        ]
        for i, text in enumerate(instructions):
            cv2.putText(
                display, text, (12, 24 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 0), 1, cv2.LINE_AA
            )

        cv2.putText(
            display, f"Selected points: {len(self.points)}",
            (WINDOW_WIDTH - 270, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        cv2.putText(
            display, f"Zoom: {self.zoom:.2f}x | Radius: {NEIGHBORHOOD_RADIUS}",
            (WINDOW_WIDTH - 330, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1
        )
        return display


# ============================================================
# BUILD THE NEIGHBOURHOOD OF ALL CLICKED POINTS
# ============================================================

def make_selected_mask(shape, points, radius):
    height, width = shape
    selected = np.zeros((height, width), dtype=bool)

    for x, y in points:
        x1 = max(0, x - radius)
        x2 = min(width, x + radius + 1)
        y1 = max(0, y - radius)
        y2 = min(height, y + radius + 1)
        selected[y1:y2, x1:x2] = True

    return selected


# ============================================================
# CHECK SATURATION AND CALCULATE RAW CFA SIGNALS
# ============================================================

def calculate_measurements(data, points):
    selected_mask = make_selected_mask(
        data["raw_data"].shape,
        points,
        NEIGHBORHOOD_RADIUS
    )

    raw_data = data["raw_data"]
    signal = data["corrected_signal"]
    not_saturated = data["not_saturated"]
    red_mask = data["red_mask"]
    green_mask = data["green_mask"]
    blue_mask = data["blue_mask"]

    # Saturation check across ALL CFA colours in the selected neighbourhood.
    selected_raw = raw_data[selected_mask]
    selected_threshold = data["saturation_threshold"][selected_mask]
    saturated_any = selected_raw >= selected_threshold
    saturated_count = int(np.count_nonzero(saturated_any))
    selected_count = int(selected_raw.size)
    saturated_fraction = saturated_count / selected_count if selected_count else 0.0

    # Per-colour counts and usable samples after the near-saturation check.
    channel_masks = {
        "red": selected_mask & red_mask,
        "green": selected_mask & green_mask,
        "blue": selected_mask & blue_mask,
    }
    channel_stats = {}
    for name, mask in channel_masks.items():
        valid = mask & not_saturated
        total = int(np.count_nonzero(mask))
        used = int(np.count_nonzero(valid))
        channel_stats[name] = {
            "mask": mask,
            "valid": valid,
            "total": total,
            "used": used,
            "excluded": total - used,
        }

    result = {
        "saturated_any": bool(saturated_count > 0),
        "saturated_count": saturated_count,
        "selected_sensor_samples": selected_count,
        "saturated_fraction": saturated_fraction,
    }
    for name, stats in channel_stats.items():
        result[f"{name}_total"] = stats["total"]
        result[f"{name}_used"] = stats["used"]
        result[f"{name}_excluded"] = stats["excluded"]
        result[f"mean_{name}"] = None

    result["ratio_rg"] = None
    result["ratio_br"] = None
    result["ratio_gb"] = None

    # Keep the existing conservative policy: reject the complete location if
    # any CFA sample in the selected region reaches the configured threshold.
    if result["saturated_any"]:
        return result

    for name, stats in channel_stats.items():
        if stats["used"] == 0:
            raise ValueError(f"No usable {name.upper()} CFA samples in this selection.")
        result[f"mean_{name}"] = float(np.mean(signal[stats["valid"]]))

    mean_red = result["mean_red"]
    mean_green = result["mean_green"]
    mean_blue = result["mean_blue"]

    # Ratios are undefined if their denominator is zero or negative.
    if mean_green <= 0:
        raise ValueError("Mean green signal is zero; R/G is undefined.")
    if mean_red <= 0:
        raise ValueError("Mean red signal is zero; B/R is undefined.")
    if mean_blue <= 0:
        raise ValueError("Mean blue signal is zero; G/B is undefined.")

    result["ratio_rg"] = mean_red / mean_green
    result["ratio_br"] = mean_blue / mean_red
    result["ratio_gb"] = mean_green / mean_blue
    return result


# ============================================================
# PROCESS IMAGES
# ============================================================

csv_columns = [
    "filename",
    "measurement_number",
    "thermocouple_temperature_K",
    "measurement_status",
    "clicked_points",
    "neighborhood_radius_pixels",
    "mean_red_CFA_corrected",
    "mean_green_CFA_corrected",
    "mean_blue_CFA_corrected",
    "red_green_ratio_R_over_G",
    "blue_red_ratio_B_over_R",
    "green_blue_ratio_G_over_B",
    "selected_sensor_samples",
    "near_saturated_samples",
    "near_saturated_fraction",
    "red_samples_total",
    "green_samples_total",
    "blue_samples_total",
    "red_samples_used",
    "green_samples_used",
    "blue_samples_used",
    "red_samples_excluded_saturation",
    "green_samples_excluded_saturation",
    "blue_samples_excluded_saturation",
    "saturation_fraction_threshold",
    "reported_white_level"
]

dataset = []
quit_program = False

for image_number, file_path in enumerate(file_paths, start=1):
    print("\n" + "=" * 65)
    print(f"IMAGE {image_number}/{len(file_paths)}")
    print(os.path.basename(file_path))
    print("=" * 65)

    try:
        data = load_raw_image(file_path)
    except Exception as error:
        print("Could not load image:", error)
        continue

    selector = MouseSelector(data["preview_rgb"])
    measurement_number = 0

    print("\nControls:")
    print("  Left click  : select flame points around bead")
    print("  Mouse wheel : zoom")
    print("  Right-drag  : pan")
    print("  T           : finish measurement location")
    print("  N           : finish image")
    print("  Q / ESC     : quit")

    while True:
        cv2.imshow(WINDOW_NAME, selector.render())
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("t"), ord("T")):
            if not selector.points:
                print("No points selected.")
                continue

            points = selector.points.copy()

            try:
                measurements = calculate_measurements(data, points)
            except Exception as error:
                print("Measurement error:", error)
                continue

            print("\nMeasurement check")
            print("-" * 45)
            print("Selected points:", len(points))
            print("Sensor samples checked:", measurements["selected_sensor_samples"])
            print("Saturated/near-saturated samples:", measurements["saturated_count"])
            print("Saturated fraction:", f"{measurements['saturated_fraction']:.2%}")

            if measurements["saturated_any"]:
                print("\n*** SATURATION WARNING ***")
                print(
                    "At least one sensor sample in the selected "
                    "thermocouple neighbourhood reached the configured "
                    "saturation threshold."
                )
                print(
                    "THIS MEASUREMENT LOCATION WILL NOT BE SAVED "
                    "AS A CALIBRATION DATA POINT."
                )
                print(
                    "Retake the image with a shorter exposure and/or "
                    "lower ISO, ensuring the flame signal is not clipped."
                )
                status = "REJECTED_SATURATION"

                # Record rejection in CSV so the user can identify
                # which image/location needs to be retaken.
                temperature = None
                while True:
                    temp_text = input(
                        "Enter thermocouple temperature in K for the "
                        "rejection log (or Enter to leave blank): "
                    ).strip()
                    if temp_text == "":
                        break
                    try:
                        temperature = float(temp_text)
                        if temperature > 0:
                            break
                        print("Temperature must be positive.")
                    except ValueError:
                        print("Enter a valid number or press Enter.")

                measurement_number += 1
                dataset.append({
                    "filename": os.path.basename(file_path),
                    "measurement_number": measurement_number,
                    "thermocouple_temperature_K": (
                        temperature if temperature is not None else ""
                    ),
                    "measurement_status": status,
                    "clicked_points": len(points),
                    "neighborhood_radius_pixels": NEIGHBORHOOD_RADIUS,
                    "mean_red_CFA_corrected": "",
                    "mean_green_CFA_corrected": "",
                    "mean_blue_CFA_corrected": "",
                    "red_green_ratio_R_over_G": "",
                    "blue_red_ratio_B_over_R": "",
                    "green_blue_ratio_G_over_B": "",
                    "selected_sensor_samples": measurements["selected_sensor_samples"],
                    "near_saturated_samples": measurements["saturated_count"],
                    "near_saturated_fraction": measurements["saturated_fraction"],
                    "red_samples_total": measurements["red_total"],
                    "green_samples_total": measurements["green_total"],
                    "blue_samples_total": measurements["blue_total"],
                    "red_samples_used": measurements["red_used"],
                    "green_samples_used": measurements["green_used"],
                    "blue_samples_used": measurements["blue_used"],
                    "red_samples_excluded_saturation": measurements["red_excluded"],
                    "green_samples_excluded_saturation": measurements["green_excluded"],
                    "blue_samples_excluded_saturation": measurements["blue_excluded"],
                    "saturation_fraction_threshold": SATURATION_FRACTION,
                    "reported_white_level": data["white_level"]
                })

            else:
                print("Mean corrected red CFA signal:", measurements["mean_red"])
                print("Mean corrected green CFA signal:", measurements["mean_green"])
                print("Mean corrected blue CFA signal:", measurements["mean_blue"])
                print("R/G ratio (R divided by G):", measurements["ratio_rg"])
                print("B/R ratio (B divided by R):", measurements["ratio_br"])
                print("G/B ratio (G divided by B):", measurements["ratio_gb"])

                while True:
                    temp_text = input(
                        "Enter corresponding thermocouple temperature in K "
                        "(Enter to skip this measurement): "
                    ).strip()

                    if temp_text == "":
                        temperature = None
                        break

                    try:
                        temperature = float(temp_text)
                        if temperature <= 0:
                            print("Temperature must be positive.")
                            continue
                        break
                    except ValueError:
                        print("Enter a valid numerical temperature.")

                if temperature is not None:
                    measurement_number += 1
                    dataset.append({
                        "filename": os.path.basename(file_path),
                        "measurement_number": measurement_number,
                        "thermocouple_temperature_K": temperature,
                        "measurement_status": "ACCEPTED",
                        "clicked_points": len(points),
                        "neighborhood_radius_pixels": NEIGHBORHOOD_RADIUS,
                        "mean_red_CFA_corrected": measurements["mean_red"],
                        "mean_green_CFA_corrected": measurements["mean_green"],
                        "mean_blue_CFA_corrected": measurements["mean_blue"],
                        "red_green_ratio_R_over_G": measurements["ratio_rg"],
                        "blue_red_ratio_B_over_R": measurements["ratio_br"],
                        "green_blue_ratio_G_over_B": measurements["ratio_gb"],
                        "selected_sensor_samples": measurements["selected_sensor_samples"],
                        "near_saturated_samples": measurements["saturated_count"],
                        "near_saturated_fraction": measurements["saturated_fraction"],
                        "red_samples_total": measurements["red_total"],
                        "green_samples_total": measurements["green_total"],
                        "blue_samples_total": measurements["blue_total"],
                        "red_samples_used": measurements["red_used"],
                        "green_samples_used": measurements["green_used"],
                        "blue_samples_used": measurements["blue_used"],
                        "red_samples_excluded_saturation": measurements["red_excluded"],
                        "green_samples_excluded_saturation": measurements["green_excluded"],
                        "blue_samples_excluded_saturation": measurements["blue_excluded"],
                        "saturation_fraction_threshold": SATURATION_FRACTION,
                        "reported_white_level": data["white_level"]
                    })
                    print("Measurement added to dataset.")
                else:
                    print("Measurement skipped.")

            # Clear clicked points before selecting the next
            # thermocouple location in the same image.
            selector.points = []

        elif key in (ord("n"), ord("N")):
            if selector.points:
                print(
                    "You have unprocessed points. Press T to process them "
                    "before moving to the next image."
                )
                continue
            break

        elif key in (ord("q"), ord("Q"), 27):
            quit_program = True
            break

    cv2.destroyWindow(WINDOW_NAME)

    if quit_program:
        break


# ============================================================
# APPEND NEW ROWS TO CSV (DO NOT OVERWRITE PREVIOUS RUNS)
# ============================================================

if dataset:
    output_path = os.path.abspath(OUTPUT_CSV)

    # Write the header only if the CSV does not exist yet or is empty.
    # Append mode preserves rows saved during earlier runs of this script.
    csv_already_has_data = (
        os.path.exists(output_path) and os.path.getsize(output_path) > 0
    )

    with open(
        output_path,
        "a",
        newline="",
        encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=csv_columns,
            extrasaction="ignore"
        )
        if not csv_already_has_data:
            writer.writeheader()
        writer.writerows(dataset)

    print("\n" + "=" * 65)
    print("PROCESSING COMPLETED")
    print("=" * 65)
    print("New measurements appended this run:", len(dataset))
    print("CSV saved to:", output_path)
    print("Previous rows in this CSV have been preserved.")
    print(
        "Rows marked REJECTED_SATURATION must not be used "
        "for calibration; retake those images."
    )
else:
    print("\nNo measurements were logged; the CSV was not changed.")
