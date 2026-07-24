"""Automatic implant osteotomy geometry fitting for Geomagic Wrap 2021.

Select the complete 360-degree inner surface and run the script. It creates
an iteratively fitted center axis, a whole-wall-radius reference cylinder,
and, when detected, a separate apical cone.
"""

import builtins
import ctypes
import math
import time
import traceback

import geomagic.app.v3
from geomagic.app.v3.imports import *


PROJECT_NAME = "Implant Osteotomy Geometry Fitter"
PROJECT_VERSION = "1.0.0"


# User settings

# Uniform slices fit the axis; focused slices only resolve the apical cone.
NUM_SLICES = 21
APICAL_FOCUS_SLICES = 13
APICAL_FOCUS_FRACTION = 0.30
AUTO_ORIENTATION_SLICES = 17
AUTO_ORIENTATION_END_MARGIN_FRACTION = 0.02
AUTO_ORIENTATION_TAIL_SECTIONS = 4
AUTO_ORIENTATION_MIN_VALID_SECTIONS = 8
AUTO_ORIENTATION_MIN_RADIUS_CONTRAST = 0.03
MAX_ITERATIONS = 20
CONSECUTIVE_STABLE_PASSES_REQUIRED = 2

# Avoid the coronal edge; retain the complete apical range.
PROFILE_CORONAL_MARGIN_FRACTION = 0.05
PROFILE_APICAL_MARGIN_FRACTION = 0.00
MIN_PROFILE_SPAN_MM = 2.0

ANGLE_TOLERANCE_DEG = 0.02
SHIFT_TOLERANCE_MM = 0.01

MIN_SECTION_POINTS = 10
AXIS_SECTIONS_REQUIRE_CLOSED = True

# Wall/cone segmentation; dr/dz slopes are dimensionless.
MIN_WALL_SECTIONS = 5
MIN_CONE_SECTIONS = 3
CONE_SLOPE_RATIO = 3.0
CONE_MIN_ABS_SLOPE = 0.20
CONE_MIN_SLOPE_DIFFERENCE = 0.15
CONE_MIN_BIC_IMPROVEMENT = 6.0
CONE_TRANSITION_GUARD_SECTIONS = 1
CONE_ALLOW_SLOPE_FALLBACK = True

BOUNDARY_SEARCH_STEP_MM = 0.05
MAX_BOUNDARY_SEARCH_DISTANCE_MM = 20.0
BOUNDARY_MIN_SECTION_POINTS = 6
APEX_BISECTION_STEPS = 4

# Native fitting contributes only the apical extent.
USE_NATIVE_BEST_FIT_APICAL_EXTENT = True
NATIVE_EXTENT_OUTLIER = 0.0
NATIVE_ROBUST_FIT_ITERATIONS = 2

ADD_FINAL_SECTION_CIRCLES = True
ADD_FINAL_CENTER_POINTS = True
ADD_AXIS_CONSTRAINED_CYLINDER = True
ADD_APICAL_CONE = True

SCRIPT_VERSION = "{} / algorithm v18 (2026-07-24)".format(PROJECT_VERSION)


# Vector helpers

def xyz(v):
    return (float(v.x()), float(v.y()), float(v.z()))


def gv(a):
    return Vector3D(float(a[0]), float(a[1]), float(a[2]))


def v_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def v_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_norm(a):
    return math.sqrt(max(0.0, v_dot(a, a)))


def v_unit(a):
    n = v_norm(a)
    if n <= 1.0e-15:
        raise ValueError("Zero-length vector")
    return v_mul(a, 1.0 / n)


def v_mean(values):
    if not values:
        raise ValueError("Cannot average an empty list")
    sx = sum(p[0] for p in values)
    sy = sum(p[1] for p in values)
    sz = sum(p[2] for p in values)
    k = 1.0 / float(len(values))
    return (sx * k, sy * k, sz * k)


def point_line_distance(point, line_point, line_direction):
    d = v_unit(line_direction)
    delta = v_sub(point, line_point)
    perpendicular = v_sub(delta, v_mul(d, v_dot(delta, d)))
    return v_norm(perpendicular)


def angle_deg(a, b):
    ua = v_unit(a)
    ub = v_unit(b)
    # Geomagic's wildcard imports also expose ``abs``.
    value = max(-1.0, min(1.0, builtins.abs(v_dot(ua, ub))))
    return math.degrees(math.acos(value))


def mm(value_in_metres):
    return float(value_in_metres) * 1000.0


def fmt_point(p):
    return "({:.6f}, {:.6f}, {:.6f})".format(p[0], p[1], p[2])


# Radius-profile helpers

def numeric_median(values):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot calculate the median of an empty list")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def median_smooth_three(values):
    """Three-point median filter that keeps the first and last positions."""
    result = []
    for index in range(len(values)):
        first = max(0, index - 1)
        last = min(len(values), index + 2)
        result.append(numeric_median(values[first:last]))
    return result


def linear_profile_fit(offsets, radii):
    """Ordinary least-squares line r = intercept + slope * offset."""
    if len(offsets) != len(radii) or len(offsets) < 2:
        raise ValueError("A profile line needs at least two paired values")

    x_mean = sum(offsets) / float(len(offsets))
    y_mean = sum(radii) / float(len(radii))
    denominator = sum((x - x_mean) ** 2 for x in offsets)
    if denominator <= 1.0e-30:
        raise ValueError("Profile offsets do not span a usable distance")

    slope = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(offsets, radii)
    ) / denominator
    intercept = y_mean - slope * x_mean
    residuals = [
        y - (intercept + slope * x)
        for x, y in zip(offsets, radii)
    ]
    sse = sum(residual * residual for residual in residuals)
    return intercept, slope, sse


def classify_wall_and_cone(records):
    """Split the radius profile into slow-taper wall and steep apical cone."""
    ordered = sorted(records, key=lambda item: item["offset"])
    count = len(ordered)
    minimum_count = MIN_WALL_SECTIONS + MIN_CONE_SECTIONS
    no_split = {
        "detected": False,
        "split_offset": None,
        "wall_slope": float("nan"),
        "cone_slope": float("nan"),
        "bic_improvement": 0.0,
        "detection_method": "none",
    }
    if count < minimum_count:
        return ordered, [], no_split

    offsets = [item["offset"] for item in ordered]
    raw_radii = [item["radius"] for item in ordered]
    radii = median_smooth_three(raw_radii)

    try:
        _, _, single_sse = linear_profile_fit(offsets, radii)
    except Exception:
        return ordered, [], no_split

    epsilon = 1.0e-30
    single_bic = (
        count * math.log(max(single_sse / float(count), epsilon))
        + 2.0 * math.log(float(count))
    )
    best = None

    # split_index is the first point assigned to the cone model.
    for split_index in range(
            MIN_WALL_SECTIONS,
            count - MIN_CONE_SECTIONS + 1):
        wall_offsets = offsets[:split_index]
        wall_radii = radii[:split_index]
        cone_offsets = offsets[split_index:]
        cone_radii = radii[split_index:]

        try:
            _, wall_slope, wall_sse = linear_profile_fit(
                wall_offsets, wall_radii
            )
            _, cone_slope, cone_sse = linear_profile_fit(
                cone_offsets, cone_radii
            )
        except Exception:
            continue

        # A true apical cone must shrink faster toward +axis than the wall.
        required_cone_magnitude = max(
            CONE_MIN_ABS_SLOPE,
            CONE_SLOPE_RATIO * max(builtins.abs(wall_slope), 0.02),
        )
        if cone_slope >= wall_slope - CONE_MIN_SLOPE_DIFFERENCE:
            continue
        if builtins.abs(cone_slope) < required_cone_magnitude:
            continue
        if cone_radii[-1] >= cone_radii[0]:
            continue

        total_sse = wall_sse + cone_sse
        two_line_bic = (
            count * math.log(max(total_sse / float(count), epsilon))
            + 4.0 * math.log(float(count))
        )
        candidate = (
            two_line_bic,
            split_index,
            wall_slope,
            cone_slope,
        )
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is None:
        return ordered, [], no_split

    improvement = single_bic - best[0]
    bic_supported = improvement >= CONE_MIN_BIC_IMPROVEMENT
    if not bic_supported and not CONE_ALLOW_SLOPE_FALLBACK:
        return ordered, [], no_split

    split_index = best[1]
    guard = max(0, int(CONE_TRANSITION_GUARD_SECTIONS))
    wall_end = max(MIN_WALL_SECTIONS, split_index - guard)
    cone_start = min(count, split_index + guard)
    wall_records = ordered[:wall_end]
    cone_records = ordered[cone_start:]

    split_offset = 0.5 * (
        ordered[split_index - 1]["offset"]
        + ordered[split_index]["offset"]
    )
    diagnostic = {
        "detected": True,
        "split_offset": split_offset,
        "wall_slope": best[2],
        "cone_slope": best[3],
        "bic_improvement": improvement,
        "detection_method": (
            "two-line BIC"
            if bic_supported
            else "constrained-slope geometry fallback"
        ),
    }
    return wall_records, cone_records, diagnostic


def merge_section_records(record_groups):
    """Merge uniform/focused samples without counting duplicate planes twice."""
    combined = []
    for group in record_groups:
        combined.extend(group)

    # At a duplicated end plane, retain the uniform record so the axis fit
    # remains depth-balanced.
    priority = {"uniform": 0, "apical_focus": 1, "final_profile": 2}
    ordered = sorted(
        combined,
        key=lambda item: (
            item["offset"],
            priority.get(item.get("sampling_group"), 9),
        ),
    )
    unique = []
    tolerance = 1.0e-10
    for record in ordered:
        if (unique and
                builtins.abs(record["offset"] - unique[-1]["offset"])
                <= tolerance):
            continue
        unique.append(record)
    return unique


def depth_weighted_mean_radius(records):
    """Return the axial-depth-weighted mean wall radius."""
    ordered = sorted(records, key=lambda item: item["offset"])
    if not ordered:
        raise ValueError("No wall records are available for radius fitting")
    if len(ordered) == 1:
        return float(ordered[0]["radius"])

    integral = 0.0
    span = 0.0
    for first, second in zip(ordered[:-1], ordered[1:]):
        dz = float(second["offset"]) - float(first["offset"])
        if dz <= 1.0e-12:
            continue
        integral += 0.5 * (
            float(first["radius"]) + float(second["radius"])
        ) * dz
        span += dz
    if span <= 1.0e-12:
        return sum(
            float(item["radius"]) for item in ordered
        ) / float(len(ordered))
    return integral / span


def interpolated_profile_radius(records, target_offset):
    """Linearly interpolate a fitted-section radius at an axial offset."""
    ordered = sorted(records, key=lambda item: item["offset"])
    if not ordered:
        raise ValueError("Cannot interpolate an empty radius profile")
    if target_offset <= ordered[0]["offset"]:
        return float(ordered[0]["radius"])
    if target_offset >= ordered[-1]["offset"]:
        return float(ordered[-1]["radius"])

    for first, second in zip(ordered[:-1], ordered[1:]):
        z0 = float(first["offset"])
        z1 = float(second["offset"])
        if z0 <= target_offset <= z1:
            if z1 - z0 <= 1.0e-12:
                return 0.5 * (
                    float(first["radius"]) + float(second["radius"])
                )
            fraction = (target_offset - z0) / (z1 - z0)
            return (
                (1.0 - fraction) * float(first["radius"])
                + fraction * float(second["radius"])
            )
    return float(ordered[-1]["radius"])


def orient_endpoints_by_radius(
        endpoint_a, endpoint_b, radius_a, radius_b):
    scale = max(radius_a, radius_b, 1.0e-12)
    contrast = builtins.abs(radius_a - radius_b) / scale
    if contrast < AUTO_ORIENTATION_MIN_RADIUS_CONTRAST:
        raise RuntimeError(
            "Automatic coronal/apical orientation is ambiguous: end radii "
            "are {:.4f} and {:.4f} mm (contrast {:.2f}%; required {:.2f}%)."
            .format(
                mm(radius_a),
                mm(radius_b),
                100.0 * contrast,
                100.0 * AUTO_ORIENTATION_MIN_RADIUS_CONTRAST,
            )
        )
    if radius_a > radius_b:
        return (endpoint_a, endpoint_b), contrast
    return (endpoint_b, endpoint_a), contrast


# Wrap UI helpers

def message(text, title=PROJECT_NAME, flags=0):
    return ctypes.windll.user32.MessageBoxW(
        None, str(text), str(title), int(flags)
    )


def warning(text):
    return message(text, PROJECT_NAME, 0x00000000 | 0x00000030)


def refresh_ui():
    """Let Wrap repaint its UI while the script is working."""
    try:
        geoapp.updateGUI()
    except Exception:
        pass


def snapshot_triangle_selection(mesh, selection):
    """Copy the active triangle selection for the complete run."""
    if selection is None or int(selection.numSelected) <= 0:
        return None

    face_ids = IntArray()
    selection.getSelectedToArray(face_ids)
    copied = TriangleSelection(mesh)
    copied.setSelectedFromArray(face_ids, True)
    return copied


# Geomagic fitting helpers


def fit_circle_to_polyline(polyline, min_points=MIN_SECTION_POINTS):
    point_array = polyline.points
    if len(point_array) < int(min_points):
        return None

    data = Points()
    for point in point_array:
        data.addPoint(point)
    data.notifyPointsAddedComplete()

    selection = PointSelection(data)
    selection.selectAll()

    fitter = BestFitCircle()
    fitter.selection = selection
    fitter.solidity = BestFitCircle.Hollow
    fitter.contactFitting = False
    fitter.justBoundary = True
    fitter.robustFitIterations = 0
    fitter.run()

    circle = fitter.resultFeature
    if circle is None or float(circle.radius) <= 1.0e-12:
        return None
    return circle


def best_circle_from_section(
        section, expected_center, require_closed=False,
        min_points=MIN_SECTION_POINTS):
    """Choose the fitted circle nearest the current axis."""
    candidates = []
    max_index = int(section.maxPolylineIndex)

    for index in range(max_index + 1):
        if not section.isValidPolylineIndex(index):
            continue
        polyline = section.getPolyline(index)
        if polyline is None or len(polyline.points) < int(min_points):
            continue
        if require_closed and not bool(polyline.isClosed):
            continue
        try:
            circle = fit_circle_to_polyline(
                polyline, min_points=min_points
            )
        except Exception:
            continue
        if circle is None:
            continue

        center = xyz(circle.center)
        radius = float(circle.radius)
        center_offset = v_norm(v_sub(center, expected_center))
        closed_penalty = 0.0 if bool(polyline.isClosed) else 10.0
        score = center_offset / max(radius, 1.0e-12) + closed_penalty
        candidates.append((score, -len(polyline.points), circle, polyline))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def fit_native_extent_cylinder(triangle_selection):
    """Run Wrap's native cylinder fit for endpoint/extent information only."""
    if triangle_selection is None or int(triangle_selection.numSelected) <= 0:
        raise ValueError(
            "A non-empty triangle selection is required for native extent fit"
        )

    fitter = BestFitCylinder()
    fitter.selection = triangle_selection
    fitter.contactFitting = False
    fitter.extentOutlier = float(NATIVE_EXTENT_OUTLIER)
    fitter.robustFitIterations = int(NATIVE_ROBUST_FIT_ITERATIONS)
    fitter.run()

    cylinder = fitter.resultFeature
    if cylinder is None:
        raise RuntimeError("Wrap BestFitCylinder returned no result feature")
    return cylinder


def native_apical_extent_on_wall_axis(
        triangle_selection, wall_midpoint, wall_direction,
        native_cylinder=None):
    """Project the native fit's apical endpoint onto the final wall axis."""
    if native_cylinder is None:
        native_cylinder = fit_native_extent_cylinder(triangle_selection)
    endpoint_a = xyz(native_cylinder.centerFrom)
    endpoint_b = xyz(native_cylinder.centerTo)

    offset_a = v_dot(
        v_sub(endpoint_a, wall_midpoint), wall_direction
    )
    offset_b = v_dot(
        v_sub(endpoint_b, wall_midpoint), wall_direction
    )
    if offset_a >= offset_b:
        apical_endpoint = endpoint_a
        apical_offset = offset_a
        coronal_endpoint = endpoint_b
        coronal_offset = offset_b
    else:
        apical_endpoint = endpoint_b
        apical_offset = offset_b
        coronal_endpoint = endpoint_a
        coronal_offset = offset_a

    projected_apical_center = v_add(
        wall_midpoint, v_mul(wall_direction, apical_offset)
    )
    lateral_distance = v_norm(
        v_sub(apical_endpoint, projected_apical_center)
    )
    return {
        "native_cylinder": native_cylinder,
        "native_apical_endpoint": apical_endpoint,
        "native_coronal_endpoint": coronal_endpoint,
        "native_apical_offset": apical_offset,
        "native_coronal_offset": coronal_offset,
        "projected_apical_center": projected_apical_center,
        "apical_lateral_distance": lateral_distance,
    }


def profile_sampling_offsets(
        native_cylinder, axis_midpoint, axis_direction):
    """Geometry-defined interval with no apical trimming."""
    endpoint_a = xyz(native_cylinder.centerFrom)
    endpoint_b = xyz(native_cylinder.centerTo)
    projected = sorted([
        v_dot(v_sub(endpoint_a, axis_midpoint), axis_direction),
        v_dot(v_sub(endpoint_b, axis_midpoint), axis_direction),
    ])
    full_start = projected[0]
    full_end = projected[1]
    full_span = full_end - full_start
    if full_span < MIN_PROFILE_SPAN_MM / 1000.0:
        raise RuntimeError(
            "The selected-surface native extent spans only {:.4f} mm along "
            "the current axis; at least {:.4f} mm is required.".format(
                mm(full_span), MIN_PROFILE_SPAN_MM
            )
        )

    coronal_margin_fraction = max(
        0.0, min(0.45, float(PROFILE_CORONAL_MARGIN_FRACTION))
    )
    apical_margin_fraction = max(
        0.0, min(0.45, float(PROFILE_APICAL_MARGIN_FRACTION))
    )
    sample_start = full_start + coronal_margin_fraction * full_span
    sample_end = full_end - apical_margin_fraction * full_span
    sample_span = sample_end - sample_start
    focus_fraction = max(
        0.05, min(0.80, float(APICAL_FOCUS_FRACTION))
    )
    return {
        "full_start": full_start,
        "full_end": full_end,
        "sample_start": sample_start,
        "sample_end": sample_end,
        "focus_start": sample_end - focus_fraction * sample_span,
        "full_span": full_span,
    }


def section_records(mesh, triangle_selection, axis_midpoint,
                    axis_direction, offset_start, offset_end,
                    number_of_slices, min_points=MIN_SECTION_POINTS,
                    sampling_group="uniform"):
    records = []
    failures = []

    if number_of_slices == 1:
        offsets = [0.5 * (offset_start + offset_end)]
    else:
        step = (
            float(offset_end) - float(offset_start)
        ) / float(number_of_slices - 1)
        offsets = [
            float(offset_start) + step * i
            for i in range(number_of_slices)
        ]

    for slice_index, offset in enumerate(offsets):
        print(
            "  Processing section {}/{}...".format(
                slice_index + 1, len(offsets)
            )
        )
        refresh_ui()
        plane_origin = v_add(axis_midpoint, v_mul(axis_direction, offset))
        plane = Plane()
        plane.initialize(gv(plane_origin), gv(axis_direction))

        # Use the construction pattern from the Wrap 2021 local API example.
        sectioner = SectionByPlane()
        sectioner.mesh = mesh
        sectioner.plane = plane
        sectioner.mergeEnds = True
        if triangle_selection is not None:
            sectioner.selection = triangle_selection

        try:
            print("    intersecting mesh with plane...")
            refresh_ui()
            sectioner.run()
            print("    intersection complete; fitting circle...")
            refresh_ui()
            section = sectioner.section
            circle = best_circle_from_section(
                section,
                plane_origin,
                require_closed=AXIS_SECTIONS_REQUIRE_CLOSED,
                min_points=min_points,
            )
            print("    circle fit complete")
            refresh_ui()
        except Exception as exc:
            failures.append((slice_index + 1, str(exc)))
            continue

        if circle is None:
            failures.append((slice_index + 1, "No usable closed section"))
            continue
        records.append({
            "offset": offset,
            "circle": circle,
            "center": xyz(circle.center),
            "radius": float(circle.radius),
            "slice_index": slice_index + 1,
            "sampling_group": sampling_group,
        })

    return records, failures


def collect_profile_records(
        mesh, triangle_selection, axis_midpoint, axis_direction, sampling):
    """Collect depth-balanced wall samples plus a dense apical diagnostic set."""
    uniform_records, uniform_failures = section_records(
        mesh,
        triangle_selection,
        axis_midpoint,
        axis_direction,
        sampling["sample_start"],
        sampling["sample_end"],
        NUM_SLICES,
        min_points=MIN_SECTION_POINTS,
        sampling_group="uniform",
    )
    focused_records, focused_failures = section_records(
        mesh,
        triangle_selection,
        axis_midpoint,
        axis_direction,
        sampling["focus_start"],
        sampling["sample_end"],
        APICAL_FOCUS_SLICES,
        min_points=BOUNDARY_MIN_SECTION_POINTS,
        sampling_group="apical_focus",
    )
    records = merge_section_records([uniform_records, focused_records])
    failures = [
        ("uniform", item[0], item[1]) for item in uniform_failures
    ] + [
        ("apical_focus", item[0], item[1]) for item in focused_failures
    ]
    return records, failures


def automatic_initial_endpoints(
        mesh, triangle_selection, native_cylinder):
    """Orient the native cylinder axis from larger coronal to smaller apical end."""
    endpoint_a = xyz(native_cylinder.centerFrom)
    endpoint_b = xyz(native_cylinder.centerTo)
    span = v_norm(v_sub(endpoint_b, endpoint_a))
    if span < MIN_PROFILE_SPAN_MM / 1000.0:
        raise RuntimeError(
            "Native seed axis spans only {:.4f} mm.".format(mm(span))
        )

    midpoint = v_mul(v_add(endpoint_a, endpoint_b), 0.5)
    direction = v_unit(v_sub(endpoint_b, endpoint_a))
    margin_fraction = max(
        0.0, min(0.20, float(AUTO_ORIENTATION_END_MARGIN_FRACTION))
    )
    margin = margin_fraction * span
    offset_start = -0.5 * span + margin
    offset_end = 0.5 * span - margin

    records, failures = section_records(
        mesh,
        triangle_selection,
        midpoint,
        direction,
        offset_start,
        offset_end,
        AUTO_ORIENTATION_SLICES,
        min_points=BOUNDARY_MIN_SECTION_POINTS,
        sampling_group="orientation",
    )
    ordered = sorted(records, key=lambda item: item["offset"])
    if len(ordered) < AUTO_ORIENTATION_MIN_VALID_SECTIONS:
        raise RuntimeError(
            "Automatic axis orientation found only {} valid sections "
            "({} required; {} failed). Select the complete inner wall, "
            "including the apical cone.".format(
                len(ordered),
                AUTO_ORIENTATION_MIN_VALID_SECTIONS,
                len(failures),
            )
        )

    tail_count = min(
        int(AUTO_ORIENTATION_TAIL_SECTIONS),
        len(ordered) // 2,
    )
    if tail_count < 2:
        raise RuntimeError(
            "Too few end sections to determine coronal/apical direction."
        )

    first_radius = numeric_median([
        item["radius"] for item in ordered[:tail_count]
    ])
    last_radius = numeric_median([
        item["radius"] for item in ordered[-tail_count:]
    ])
    _, profile_slope, _ = linear_profile_fit(
        [item["offset"] for item in ordered],
        median_smooth_three([item["radius"] for item in ordered]),
    )
    endpoints, contrast = orient_endpoints_by_radius(
        endpoint_a,
        endpoint_b,
        first_radius,
        last_radius,
    )

    diagnostic = {
        "valid_sections": len(ordered),
        "first_radius": first_radius,
        "last_radius": last_radius,
        "relative_contrast": contrast,
        "profile_slope": profile_slope,
    }
    return endpoints, diagnostic


def closed_circle_at_axis_offset(
        mesh, triangle_selection, axis_midpoint, axis_direction, offset,
        min_points=MIN_SECTION_POINTS):
    """Create one strict closed section at an offset on the locked axis."""
    plane_origin = v_add(axis_midpoint, v_mul(axis_direction, offset))
    plane = Plane()
    plane.initialize(gv(plane_origin), gv(axis_direction))

    sectioner = SectionByPlane()
    sectioner.mesh = mesh
    sectioner.plane = plane
    sectioner.mergeEnds = True
    if triangle_selection is not None:
        sectioner.selection = triangle_selection

    try:
        sectioner.run()
        circle = best_circle_from_section(
            sectioner.section,
            plane_origin,
            require_closed=True,
            min_points=min_points,
        )
    except Exception as exc:
        return None, str(exc)

    if circle is None:
        return None, "No closed section"
    return circle, None


def search_closed_boundary(mesh, triangle_selection, axis_midpoint,
                           axis_direction, sign, center_circle):
    """Return the last-valid/first-invalid bracket on one side of the axis."""
    step = BOUNDARY_SEARCH_STEP_MM / 1000.0
    max_distance = MAX_BOUNDARY_SEARCH_DISTANCE_MM / 1000.0
    if step <= 0.0:
        raise ValueError("BOUNDARY_SEARCH_STEP_MM must be positive")
    max_steps = int(math.floor(max_distance / step))
    if max_steps < 1:
        raise ValueError("MAX_BOUNDARY_SEARCH_DISTANCE_MM is too small")

    side_name = "+axis" if sign > 0.0 else "-axis"
    last_offset = 0.0
    last_circle = center_circle

    for step_index in range(1, max_steps + 1):
        offset = sign * step * float(step_index)
        circle, failure = closed_circle_at_axis_offset(
            mesh,
            triangle_selection,
            axis_midpoint,
            axis_direction,
            offset,
            min_points=BOUNDARY_MIN_SECTION_POINTS,
        )
        if circle is None:
            print(
                "Boundary {}: first invalid section at {:.3f} mm; "
                "previous valid section at {:.3f} mm ({})".format(
                    side_name,
                    mm(offset),
                    mm(last_offset),
                    failure,
                )
            )
            refresh_ui()
            return {
                "last_valid_offset": last_offset,
                "last_valid_circle": last_circle,
                "first_invalid_offset": offset,
                "failure": failure,
            }

        last_offset = offset
        last_circle = circle
        if step_index == 1 or step_index % 10 == 0:
            print(
                "Boundary {}: closed through {:.3f} mm".format(
                    side_name, mm(offset)
                )
            )
            refresh_ui()

    raise RuntimeError(
        "No open/non-closed boundary was found on {} within {:.3f} mm. "
        "Increase MAX_BOUNDARY_SEARCH_DISTANCE_MM or check the selection."
        .format(side_name, MAX_BOUNDARY_SEARCH_DISTANCE_MM)
    )


def refine_closed_boundary(
        mesh, triangle_selection, axis_midpoint, axis_direction,
        valid_offset, valid_circle, invalid_offset):
    """Bisect a closed/non-closed section bracket and estimate its limit."""
    last_valid_offset = float(valid_offset)
    last_valid_circle = valid_circle
    first_invalid_offset = float(invalid_offset)

    for refinement_index in range(APEX_BISECTION_STEPS):
        probe_offset = 0.5 * (
            last_valid_offset + first_invalid_offset
        )
        circle, failure = closed_circle_at_axis_offset(
            mesh,
            triangle_selection,
            axis_midpoint,
            axis_direction,
            probe_offset,
            min_points=BOUNDARY_MIN_SECTION_POINTS,
        )
        if circle is None:
            first_invalid_offset = probe_offset
            state = "invalid"
        else:
            last_valid_offset = probe_offset
            last_valid_circle = circle
            state = "closed"

        print(
            "Apex refinement {}/{}: {:.6f} mm -> {}".format(
                refinement_index + 1,
                APEX_BISECTION_STEPS,
                mm(probe_offset),
                state,
            )
        )
        refresh_ui()

    estimated_limit = 0.5 * (
        last_valid_offset + first_invalid_offset
    )
    return {
        "estimated_limit_offset": estimated_limit,
        "last_valid_offset": last_valid_offset,
        "last_valid_circle": last_valid_circle,
        "first_invalid_offset": first_invalid_offset,
        "bracket_width": builtins.abs(
            first_invalid_offset - last_valid_offset
        ),
    }


def fit_line_direction(centers, reference_direction=None):
    """Fit a deterministic equal-weight PCA line through section centres."""
    if len(centers) < 2:
        raise ValueError("At least two centers are required for line fitting")

    mean = v_mean(centers)
    covariance = [[0.0, 0.0, 0.0] for _ in range(3)]
    for center in centers:
        q = v_sub(center, mean)
        covariance[0][0] += q[0] * q[0]
        covariance[0][1] += q[0] * q[1]
        covariance[0][2] += q[0] * q[2]
        covariance[1][0] += q[1] * q[0]
        covariance[1][1] += q[1] * q[1]
        covariance[1][2] += q[1] * q[2]
        covariance[2][0] += q[2] * q[0]
        covariance[2][1] += q[2] * q[1]
        covariance[2][2] += q[2] * q[2]

    if reference_direction is not None:
        direction = v_unit(reference_direction)
    else:
        direction = v_unit(v_sub(centers[-1], centers[0]))

    for _ in range(100):
        updated = (
            covariance[0][0] * direction[0]
            + covariance[0][1] * direction[1]
            + covariance[0][2] * direction[2],
            covariance[1][0] * direction[0]
            + covariance[1][1] * direction[1]
            + covariance[1][2] * direction[2],
            covariance[2][0] * direction[0]
            + covariance[2][1] * direction[1]
            + covariance[2][2] * direction[2],
        )
        if v_norm(updated) <= 1.0e-18:
            raise RuntimeError("Section centers do not define a stable line")
        updated = v_unit(updated)
        if v_dot(updated, direction) < 0.0:
            updated = v_mul(updated, -1.0)
        change = angle_deg(direction, updated)
        direction = updated
        if change <= 1.0e-10:
            break

    if (reference_direction is not None and
            v_dot(direction, reference_direction) < 0.0):
        direction = v_mul(direction, -1.0)
    return direction


def set_name(obj, name):
    try:
        obj.name = name
    except Exception:
        pass


# Main routine

def run_axis_fit(
        model, mesh, triangle_selection, initial_endpoints,
        native_extent_cylinder=None):
    initial_start = initial_endpoints[0]
    initial_end = initial_endpoints[1]
    full_length = v_norm(v_sub(initial_end, initial_start))
    if full_length <= 1.0e-12:
        raise ValueError("The automatic native seed axis has zero length")

    current_direction = v_unit(v_sub(initial_end, initial_start))
    initial_direction = current_direction
    current_midpoint = v_mul(v_add(initial_start, initial_end), 0.5)

    print(
        "Automatic native seed span: {:.4f} mm.".format(mm(full_length))
    )
    if native_extent_cylinder is None:
        native_extent_cylinder = fit_native_extent_cylinder(
            triangle_selection
        )

    shift_tolerance = SHIFT_TOLERANCE_MM / 1000.0
    history = []
    converged = False
    stable_passes = 0
    latest_records = []
    latest_wall_records = []
    latest_cone_records = []
    latest_axis_wall_records = []
    latest_profile = None

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(
            "Starting iteration {}/{}...".format(iteration, MAX_ITERATIONS)
        )
        refresh_ui()
        sampling = profile_sampling_offsets(
            native_extent_cylinder,
            current_midpoint,
            current_direction,
        )
        print(
            "Iteration {} geometry-defined profile span: {:.4f} to "
            "{:.4f} mm (full projected span {:.4f} mm)".format(
                iteration,
                mm(sampling["sample_start"]),
                mm(sampling["sample_end"]),
                mm(sampling["full_span"]),
            )
        )
        records, _ = collect_profile_records(
            mesh,
            triangle_selection,
            current_midpoint,
            current_direction,
            sampling,
        )
        wall_records, cone_records, profile = classify_wall_and_cone(
            records
        )
        latest_records = records
        latest_wall_records = wall_records
        latest_cone_records = cone_records
        latest_profile = profile

        axis_wall_records = [
            item for item in wall_records
            if item.get("sampling_group") == "uniform"
        ]
        if len(axis_wall_records) < MIN_WALL_SECTIONS:
            # Fallback for sparse or damaged uniform sections.
            axis_wall_records = wall_records
            print(
                "Uniform wall sections were sparse; using all wall sections "
                "for this PCA update."
            )
        centers = [item["center"] for item in axis_wall_records]
        if len(centers) < MIN_WALL_SECTIONS:
            raise RuntimeError(
                "Only {} usable wall sections were found; at least {} are "
                "required. Reposition/lengthen the coronal-to-apical initial "
                "line or improve the complete inner-surface selection."
                .format(len(centers), MIN_WALL_SECTIONS)
            )
        latest_axis_wall_records = axis_wall_records

        wall_ids = set(id(item) for item in wall_records)
        cone_ids = set(id(item) for item in cone_records)
        for record in records:
            if id(record) in wall_ids:
                section_class = "WALL"
            elif id(record) in cone_ids:
                section_class = "CONE"
            else:
                section_class = "TRANSITION"
            print(
                "Iteration {} section {:02d}: offset={:.4f} mm, "
                "radius={:.6f} mm, class={}, center={}"
                .format(
                    iteration,
                    record["slice_index"],
                    mm(record["offset"]),
                    mm(record["radius"]),
                    section_class,
                    fmt_point(record["center"]),
                )
            )

        if profile["detected"]:
            print(
                "Iteration {} wall/cone change: {:.4f} mm; "
                "wall slope={:.6f}, cone slope={:.6f}, "
                "BIC improvement={:.3f}, method={}".format(
                    iteration,
                    mm(profile["split_offset"]),
                    profile["wall_slope"],
                    profile["cone_slope"],
                    profile["bic_improvement"],
                    profile["detection_method"],
                )
            )
        else:
            print(
                "Iteration {}: no slope-plausible cone segment was present "
                "in the sampled span; all {} valid sections are treated as "
                "wall.".format(
                    iteration, len(wall_records)
                )
            )

        new_midpoint = v_mean(centers)
        new_direction = fit_line_direction(
            centers, reference_direction=current_direction
        )
        if v_dot(new_direction, current_direction) < 0.0:
            new_direction = v_mul(new_direction, -1.0)

        angular_change = angle_deg(current_direction, new_direction)
        transverse_shift = point_line_distance(
            new_midpoint, current_midpoint, current_direction
        )
        centerline_rms = math.sqrt(
            sum(
                point_line_distance(c, new_midpoint, new_direction) ** 2
                for c in centers
            ) / float(len(centers))
        )
        within_tolerance = (
            angular_change <= ANGLE_TOLERANCE_DEG
            and transverse_shift <= shift_tolerance
        )
        if within_tolerance:
            stable_passes += 1
        else:
            stable_passes = 0

        history.append({
            "iteration": iteration,
            "valid_sections": len(records),
            "wall_sections": len(axis_wall_records),
            "cone_sections": len(cone_records),
            "cone_detected": profile["detected"],
            "angle_change_deg": angular_change,
            "transverse_shift_mm": mm(transverse_shift),
            "centerline_rms_mm": mm(centerline_rms),
            "within_tolerance": within_tolerance,
            "consecutive_stable_passes": stable_passes,
            "direction": new_direction,
            "midpoint": new_midpoint,
        })

        print(
            "Iteration {}: valid={}, wall-used={}, cone-excluded={}, "
            "angle change={:.6f} deg, shift={:.6f} mm, "
            "wall-center RMS={:.6f} mm, stable passes={}/{}".format(
                iteration,
                len(records),
                len(axis_wall_records),
                len(cone_records),
                angular_change,
                mm(transverse_shift),
                mm(centerline_rms),
                stable_passes,
                CONSECUTIVE_STABLE_PASSES_REQUIRED,
            )
        )

        current_midpoint = new_midpoint
        current_direction = new_direction

        if stable_passes >= CONSECUTIVE_STABLE_PASSES_REQUIRED:
            converged = True
            break
        if within_tolerance:
            print(
                "Tolerance met once; resampling the updated axis for "
                "confirmation."
            )

    # The last loop pass is also the final verified update.
    verification_records = latest_records
    verification_wall_records = latest_wall_records
    verification_cone_records = latest_cone_records
    verification_axis_wall_records = latest_axis_wall_records
    verification_profile = latest_profile
    verification_centers = [
        item["center"] for item in verification_axis_wall_records
    ]
    if history:
        final_update_angle = history[-1]["angle_change_deg"]
        final_update_shift_mm = history[-1]["transverse_shift_mm"]
    else:
        final_update_angle = float("nan")
        final_update_shift_mm = float("nan")

    # Negative direction is coronal; positive direction is apical.
    print(
        "Final wall axis locked; searching coronal boundary and apical tip..."
    )
    refresh_ui()

    center_circle, center_failure = closed_circle_at_axis_offset(
        mesh,
        triangle_selection,
        current_midpoint,
        current_direction,
        0.0,
        min_points=BOUNDARY_MIN_SECTION_POINTS,
    )
    if center_circle is None:
        raise RuntimeError(
            "The locked-axis midpoint did not produce a closed section: {}"
            .format(center_failure)
        )

    coronal_boundary = search_closed_boundary(
        mesh,
        triangle_selection,
        current_midpoint,
        current_direction,
        -1.0,
        center_circle,
    )
    apical_boundary = search_closed_boundary(
        mesh,
        triangle_selection,
        current_midpoint,
        current_direction,
        1.0,
        center_circle,
    )

    coronal_offset = coronal_boundary["last_valid_offset"]
    coronal_circle = coronal_boundary["last_valid_circle"]
    apical_last_offset = apical_boundary["last_valid_offset"]
    apical_last_circle = apical_boundary["last_valid_circle"]

    print(
        "Refining the apical last-closed/first-invalid bracket..."
    )
    apex_result = refine_closed_boundary(
        mesh,
        triangle_selection,
        current_midpoint,
        current_direction,
        apical_last_offset,
        apical_last_circle,
        apical_boundary["first_invalid_offset"],
    )
    closure_limit_offset = apex_result["estimated_limit_offset"]
    apical_last_offset = apex_result["last_valid_offset"]
    apical_last_circle = apex_result["last_valid_circle"]

    # Final locked-axis profile determines wall radius and cone transition.
    final_profile_span = apical_last_offset - coronal_offset
    if final_profile_span < MIN_PROFILE_SPAN_MM / 1000.0:
        raise RuntimeError(
            "The final closed-section profile spans only {:.4f} mm."
            .format(mm(final_profile_span))
        )
    final_sample_start = coronal_offset + min(
        BOUNDARY_SEARCH_STEP_MM / 1000.0,
        0.02 * final_profile_span,
    )
    final_sample_end = apical_last_offset
    final_focus_fraction = max(
        0.05, min(0.80, float(APICAL_FOCUS_FRACTION))
    )
    final_sampling = {
        "sample_start": final_sample_start,
        "sample_end": final_sample_end,
        "focus_start": (
            final_sample_end
            - final_focus_fraction * (final_sample_end - final_sample_start)
        ),
    }
    print(
        "Sampling locked-axis full profile from {:.4f} to {:.4f} mm; "
        "apical focus begins at {:.4f} mm.".format(
            mm(final_sampling["sample_start"]),
            mm(final_sampling["sample_end"]),
            mm(final_sampling["focus_start"]),
        )
    )
    final_profile_records, _ = collect_profile_records(
        mesh,
        triangle_selection,
        current_midpoint,
        current_direction,
        final_sampling,
    )
    final_wall_records, final_cone_records, final_profile = (
        classify_wall_and_cone(final_profile_records)
    )

    # Reuse the verified split if the final full-span profile is sparse.
    if (not final_profile["detected"] and
            verification_profile["detected"]):
        print(
            "Final full-span profile was too sparse for a cone split; "
            "reusing the locked-axis verification split."
        )
        final_profile_records = verification_records
        final_wall_records = verification_wall_records
        final_cone_records = verification_cone_records
        final_profile = verification_profile

    if len(final_wall_records) < MIN_WALL_SECTIONS:
        raise RuntimeError(
            "Only {} final wall sections remained after cone separation; "
            "at least {} are required.".format(
                len(final_wall_records), MIN_WALL_SECTIONS
            )
        )

    cylinder_radius = depth_weighted_mean_radius(final_wall_records)
    cone_base_offset = None
    cone_base_circle = None
    observed_cone_base_radius = None
    if final_profile["detected"]:
        cone_base_offset = float(final_profile["split_offset"])
        cone_base_circle, cone_base_failure = closed_circle_at_axis_offset(
            mesh,
            triangle_selection,
            current_midpoint,
            current_direction,
            cone_base_offset,
            min_points=BOUNDARY_MIN_SECTION_POINTS,
        )
        if cone_base_circle is not None:
            observed_cone_base_radius = float(cone_base_circle.radius)
        else:
            observed_cone_base_radius = interpolated_profile_radius(
                final_profile_records, cone_base_offset
            )
            print(
                "Cone-base plane did not yield a strict closed circle ({}); "
                "using profile interpolation for radius.".format(
                    cone_base_failure
                )
            )

    # Closed-section loss is a fallback when the selected surface is open.
    native_extent = None
    apex_offset = closure_limit_offset
    apex_source = "closed-section fallback"
    if USE_NATIVE_BEST_FIT_APICAL_EXTENT:
        try:
            print(
                "Reusing native BestFitCylinder for apical extent only..."
            )
            refresh_ui()
            native_extent = native_apical_extent_on_wall_axis(
                triangle_selection,
                current_midpoint,
                current_direction,
                native_cylinder=native_extent_cylinder,
            )
            native_offset = native_extent["native_apical_offset"]
            if native_offset <= apical_last_offset:
                raise RuntimeError(
                    "Native apical extent ({:.4f} mm) did not extend beyond "
                    "the last closed apical section ({:.4f} mm).".format(
                        mm(native_offset), mm(apical_last_offset)
                    )
                )
            if native_offset >= closure_limit_offset:
                apex_offset = native_offset
                apex_source = "Wrap BestFitCylinder apical extent"
            else:
                apex_offset = closure_limit_offset
                apex_source = (
                    "closed-section limit (deeper than native extent)"
                )
            print(
                "Native apical endpoint: {}; projected wall-axis offset: "
                "{:.6f} mm; lateral axis difference: {:.6f} mm".format(
                    fmt_point(native_extent["native_apical_endpoint"]),
                    mm(apex_offset),
                    mm(native_extent["apical_lateral_distance"]),
                )
            )
        except Exception as native_exc:
            native_extent = None
            print(
                "Native extent fit failed; using closed-section fallback: {}"
                .format(str(native_exc))
            )
            refresh_ui()

    cone_geometry_valid = (
        cone_base_offset is not None
        and observed_cone_base_radius is not None
        and cylinder_radius > 1.0e-12
        and cone_base_offset > coronal_offset
        and cone_base_offset < apex_offset
    )
    # Radius, cone transition and cylinder extent are independent.
    cylinder_end_offset = apex_offset
    cylinder_height = cylinder_end_offset - coronal_offset
    if cylinder_height <= 1.0e-12:
        raise RuntimeError(
            "Coronal-to-apex search produced a zero-length interval. "
            "Automatic coronal/apical orientation may be invalid."
        )
    cone_height = (
        apex_offset - cone_base_offset if cone_geometry_valid else 0.0
    )

    coronal_radius = float(coronal_circle.radius)
    apical_last_radius = float(apical_last_circle.radius)
    if coronal_radius <= apical_last_radius:
        raise RuntimeError(
            "The -axis boundary is not larger than the last apical section. "
            "Automatic coronal/apical orientation may be invalid."
        )

    cylinder_from = v_add(
        current_midpoint, v_mul(current_direction, coronal_offset)
    )
    apex_point = v_add(
        current_midpoint, v_mul(current_direction, apex_offset)
    )
    cylinder_to = apex_point

    final_axis = Line()
    final_axis.initialize(gv(cylinder_from), gv(apex_point))
    stamp = time.strftime("%H%M%S")
    set_name(final_axis, "OsteotomyAxis_WallAndCone_" + stamp)
    geoapp.addFeature(model, final_axis)

    constrained_cylinder = None
    if ADD_AXIS_CONSTRAINED_CYLINDER:
        constrained_cylinder = Cylinder()
        constrained_cylinder.initialize(
            gv(cylinder_from), gv(cylinder_to), cylinder_radius
        )
        set_name(
            constrained_cylinder,
            "OsteotomyReferenceCylinder_WholeWallRadius_ToApex_" + stamp,
        )
        geoapp.addFeature(model, constrained_cylinder)

    apical_cone = None
    if ADD_APICAL_CONE and cone_geometry_valid:
        cone_base_center = v_add(
            current_midpoint,
            v_mul(current_direction, cone_base_offset),
        )
        apical_cone = Cone()
        # Wrap's Cone initialize order is smaller face then larger face.
        apical_cone.initialize(
            gv(apex_point),
            gv(cone_base_center),
            0.0,
            cylinder_radius,
        )
        set_name(apical_cone, "OsteotomyApicalCone_" + stamp)
        geoapp.addFeature(model, apical_cone)

    final_circles = [coronal_circle, apical_last_circle]
    if cone_base_circle is not None:
        final_circles.append(cone_base_circle)
    if ADD_FINAL_SECTION_CIRCLES:
        set_name(coronal_circle, "CoronalRadiusCircle_" + stamp)
        set_name(
            apical_last_circle,
            "DiagnosticLastClosedApicalCircle_" + stamp,
        )
        geoapp.addFeature(model, coronal_circle)
        geoapp.addFeature(model, apical_last_circle)
        if cone_base_circle is not None:
            set_name(cone_base_circle, "WallConeTransitionCircle_" + stamp)
            geoapp.addFeature(model, cone_base_circle)

    if ADD_FINAL_CENTER_POINTS:
        landmark_points = [
            ("CoronalCircleCenter", xyz(coronal_circle.center)),
            (
                "LastClosedApicalCircleCenter",
                xyz(apical_last_circle.center),
            ),
            ("ApicalExtentOnWallAxis", apex_point),
        ]
        if cone_geometry_valid:
            landmark_points.append(
                (
                    "WallConeTransitionOnAxis",
                    v_add(
                        current_midpoint,
                        v_mul(current_direction, cone_base_offset),
                    ),
                )
            )
        if native_extent is not None:
            landmark_points.append(
                (
                    "NativeBestFitApicalEndpoint",
                    native_extent["native_apical_endpoint"],
                )
            )
        for point_name, center in landmark_points:
            point = PointFeature()
            point.initialize(gv(center))
            set_name(point, "{}_{}".format(point_name, stamp))
            geoapp.addFeature(model, point)

    geoapp.redraw(True)

    total_angle = angle_deg(initial_direction, current_direction)
    verification_rms = 0.0
    if verification_centers:
        verification_rms = math.sqrt(
            sum(
                point_line_distance(c, current_midpoint, current_direction) ** 2
                for c in verification_centers
            ) / float(len(verification_centers))
        )

    apex_source_label = {
        "Wrap BestFitCylinder apical extent": "原生轴向范围",
        "closed-section fallback": "闭合截面极限",
        "closed-section limit (deeper than native extent)": "闭合截面极限",
    }.get(apex_source, apex_source)

    report_lines = [
        "状态: {}".format("已收敛" if converged else "达到迭代上限"),
        "迭代: {} 次；连续稳定: {}/{}".format(
            len(history),
            stable_passes,
            CONSECUTIVE_STABLE_PASSES_REQUIRED,
        ),
        "原生种子轴→最终轴线: {:.4f}°".format(total_angle),
        "侧壁圆心 RMS: {:.4f} mm".format(mm(verification_rms)),
        "有效截面: 侧壁 {}，圆锥 {}".format(
            len(final_wall_records),
            len(final_cone_records),
        ),
        "圆柱半径 / 直径: {:.4f} / {:.4f} mm".format(
            mm(cylinder_radius),
            2.0 * mm(cylinder_radius),
        ),
        "圆柱高度（冠方至尖端平面）: {:.4f} mm".format(
            mm(cylinder_height)
        ),
        "窝洞最低点来源: {}".format(apex_source_label),
    ]
    if (not math.isnan(final_update_angle) and
            not math.isnan(final_update_shift_mm)):
        report_lines.append(
            "最终更新: {:.6f}° / {:.6f} mm".format(
                final_update_angle,
                final_update_shift_mm,
            )
        )
    if apical_cone is not None:
        report_lines.extend([
            "圆锥转折深度: {:.4f} mm".format(
                mm(cone_base_offset - coronal_offset)
            ),
            "圆锥高度: {:.4f} mm".format(mm(cone_height)),
            "输出: 轴线 + 全深度圆柱 + 根方圆锥",
        ])
    else:
        report_lines.append("警告: 未识别到可靠圆锥，未创建圆锥特征")
    if not converged:
        report_lines.append(
            "警告: 未达到连续两轮稳定，请检查迭代日志"
        )
    if native_extent is None:
        report_lines.append(
            "提示: 原生根方范围不可用，最低点采用闭合截面极限"
        )
    report = "\n".join(report_lines)
    print("\n" + report)
    message(report, "{} result".format(PROJECT_NAME), 0x00000040)
    return final_axis, constrained_cylinder, final_circles, history


def main():
    try:
        print("{} version: {}".format(PROJECT_NAME, SCRIPT_VERSION))
        refresh_ui()
        model = geoapp.getActiveModel()
        if model is None:
            warning("No active model. Activate the osteotomy-wall mesh and run again.")
            return

        mesh = geoapp.getMesh(model)
        if mesh is None:
            warning("The active model does not contain a polygon mesh.")
            return

        triangle_selection = geoapp.getActiveTriangleSelection(mesh)
        selection_count = 0
        if triangle_selection is not None:
            selection_count = int(triangle_selection.numSelected)

        if selection_count <= 0:
            warning(
                "No active inner-surface triangles are selected.\n\n"
                "Select the complete 360-degree osteotomy inner surface, "
                "including the apical cone, then run again. Exclude unrelated "
                "outer/tooth surfaces and gross entrance artifacts."
            )
            return
        print(
            "Using complete inner-surface selection: {} triangles".format(
                selection_count
            )
        )
        triangle_selection = snapshot_triangle_selection(
            mesh, triangle_selection
        )

        print("Fitting native cylinder for the automatic seed axis...")
        refresh_ui()
        native_extent_cylinder = fit_native_extent_cylinder(
            triangle_selection
        )
        initial_endpoints, orientation = automatic_initial_endpoints(
            mesh,
            triangle_selection,
            native_extent_cylinder,
        )

        print(
            "Automatic seed axis (coronal -> apical): {} -> {}".format(
                fmt_point(initial_endpoints[0]),
                fmt_point(initial_endpoints[1]),
            )
        )
        print(
            "Orientation evidence: valid sections={}, end radii={:.4f}/"
            "{:.4f} mm, contrast={:.2f}%, profile slope={:.6f}".format(
                orientation["valid_sections"],
                mm(orientation["first_radius"]),
                mm(orientation["last_radius"]),
                100.0 * orientation["relative_contrast"],
                orientation["profile_slope"],
            )
        )
        refresh_ui()

        run_axis_fit(
            model,
            mesh,
            triangle_selection,
            initial_endpoints,
            native_extent_cylinder=native_extent_cylinder,
        )

    except Exception as exc:
        details = traceback.format_exc()
        print(details)
        warning(
            "Stable axis/cylinder fitting failed:\n{}\n\n"
            "See the Scripting output panel for details."
            .format(str(exc))
        )


main()
