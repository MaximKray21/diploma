#!/usr/bin/env python3
"""Compare a calculated trajectory with the digitized 500 m reference table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIELDS = {
    "vr_m_s": ("radial_velocity_m_s", "Vr, m/s"),
    "vn_m_s": ("tangential_velocity_m_s", "Vn, m/s"),
    "height_km": ("altitude_m", "Height, km"),
    "range_km": ("beta_rad", "Range, km"),
    "beta_deg": ("beta_rad", "Beta, deg"),
    "mass_kg": ("mass_kg", "Mass, kg"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("reference_data/example_500m_trajectory.csv"),
    )
    parser.add_argument(
        "--solution",
        type=Path,
        default=Path("results_example_500m_160/T_2800_s/trajectory.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("comparison_example_500m"))
    return parser.parse_args()


def load_reference(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Reference table is empty: {path}")
    return rows


def interpolate_solution(
    reference: list[dict[str, str]], solution_path: Path
) -> dict[str, np.ndarray]:
    solution = np.genfromtxt(solution_path, delimiter=",", names=True)
    if solution.size < 2:
        raise ValueError(f"At least two solution rows are required: {solution_path}")
    with (solution_path.parent / "report.json").open(encoding="utf-8") as stream:
        moon_radius_m = json.load(stream)["parameters"]["moon_radius_m"]

    time_s = np.array([float(row["time_s"]) for row in reference])
    if time_s[0] < solution["time_s"][0] or time_s[-1] > solution["time_s"][-1]:
        raise ValueError("Reference times are outside the calculated trajectory interval")

    result = {"time_s": time_s}
    result["vr_m_s"] = np.interp(time_s, solution["time_s"], solution["radial_velocity_m_s"])
    result["vn_m_s"] = np.interp(time_s, solution["time_s"], solution["tangential_velocity_m_s"])
    result["height_km"] = np.interp(time_s, solution["time_s"], solution["altitude_m"]) / 1_000.0
    beta_rad = np.interp(time_s, solution["time_s"], solution["beta_rad"])
    result["range_km"] = beta_rad * moon_radius_m / 1_000.0
    result["beta_deg"] = np.degrees(beta_rad)
    result["mass_kg"] = np.interp(time_s, solution["time_s"], solution["mass_kg"])
    return result


def compare(
    reference: list[dict[str, str]], interpolated: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    comparison = {"time_s": interpolated["time_s"]}
    summary: dict[str, dict[str, float]] = {}
    for name in FIELDS:
        expected = np.array([float(row[name]) for row in reference])
        actual = interpolated[name]
        error = actual - expected
        comparison[f"reference_{name}"] = expected
        comparison[f"calculated_{name}"] = actual
        comparison[f"error_{name}"] = error
        summary[name] = {
            "max_abs_error": float(np.max(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "final_error": float(error[-1]),
        }
    return comparison, summary


def compare_switches(reference: list[dict[str, str]], solution_path: Path) -> list[dict[str, float | str]]:
    """Compare discrete-control switches with the annotated PDF-table boundaries."""

    solution = np.genfromtxt(solution_path, delimiter=",", names=True)
    active = solution["throttle"] >= 0.5
    edge_indices = np.flatnonzero(np.diff(active.astype(int)) != 0)
    calculated = []
    for index in edge_indices:
        kind = "power_to_coast" if active[index] else "coast_to_power"
        calculated.append((kind, float(solution["time_s"][index + 1])))

    expected = [
        ("power_to_coast", float(row["time_s"]))
        for row in reference
        if row["note"] == "end_of_first_powered_arc"
    ] + [
        ("coast_to_power", float(row["time_s"]))
        for row in reference
        if row["note"] == "start_of_second_powered_arc_duplicate"
    ]
    rows: list[dict[str, float | str]] = []
    for kind, reference_time in expected:
        candidates = [time for candidate_kind, time in calculated if candidate_kind == kind]
        if not candidates:
            rows.append({"kind": kind, "reference_time_s": reference_time, "calculated_time_s": "missing"})
            continue
        calculated_time = min(candidates, key=lambda time: abs(time - reference_time))
        rows.append(
            {
                "kind": kind,
                "reference_time_s": reference_time,
                "calculated_time_s": calculated_time,
                "difference_s": calculated_time - reference_time,
            }
        )
    return rows


def save_comparison(
    reference: list[dict[str, str]],
    comparison: dict[str, np.ndarray],
    summary: dict[str, dict[str, float]],
    switches: list[dict[str, float | str]],
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    names = ["source_row", "note", "time_s"]
    for name in FIELDS:
        names.extend([f"reference_{name}", f"calculated_{name}", f"error_{name}"])
    with (output / "pointwise_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for index, row in enumerate(reference):
            output_row = {"source_row": row["source_row"], "note": row["note"], "time_s": comparison["time_s"][index]}
            for name in FIELDS:
                output_row[f"reference_{name}"] = comparison[f"reference_{name}"][index]
                output_row[f"calculated_{name}"] = comparison[f"calculated_{name}"][index]
                output_row[f"error_{name}"] = comparison[f"error_{name}"][index]
            writer.writerow(output_row)
    (output / "summary.json").write_text(
        json.dumps({"state_errors": summary, "switches": switches}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def plot_errors(comparison: dict[str, np.ndarray], output: Path) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, constrained_layout=True)
    for axis, name in zip(axes.flat, FIELDS):
        axis.plot(comparison["time_s"], comparison[f"error_{name}"], marker="o", markersize=2.5)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(f"Error: {FIELDS[name][1]}")
        axis.set_ylabel("calculated − reference")
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Time, s")
    figure.savefig(output / "state_errors.png", dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    reference = load_reference(args.reference)
    interpolated = interpolate_solution(reference, args.solution)
    comparison, summary = compare(reference, interpolated)
    switches = compare_switches(reference, args.solution)
    save_comparison(reference, comparison, summary, switches, args.output)
    plot_errors(comparison, args.output)
    print(f"Compared {len(reference)} reference rows.")
    for name, metrics in summary.items():
        print(f"{name:10s} max={metrics['max_abs_error']:.8g} rmse={metrics['rmse']:.8g}")
    for item in switches:
        print(item)


if __name__ == "__main__":
    main()
