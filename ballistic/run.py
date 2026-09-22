#!/usr/bin/env python3
"""Command-line entry point for the lunar soft-landing optimizer."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from soft_landing import (
    LandingTarget,
    PhysicalParameters,
    SolverOptions,
    example_500m_parameters,
    example_500m_target,
    save_family_summary,
    save_solution,
    solve_family,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--duration", type=float, default=1800.0, help="landing time, seconds")
    mode.add_argument("--family", action="store_true", help="solve 600..3600 s with a 60 s step")
    mode.add_argument(
        "--example-500m",
        action="store_true",
        help="reproduce the 500 m endpoint from the example PDF",
    )
    parser.add_argument("--nodes", type=int, default=80, help="multiple-shooting intervals")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--initial-mass", type=float, default=15_000.0)
    parser.add_argument("--dry-mass", type=float, default=5_000.0)
    parser.add_argument("--thrust", type=float, default=60_000.0)
    parser.add_argument("--exhaust-velocity", type=float, default=3_050.0)
    parser.add_argument("--moon-radius", type=float, default=1_737_400.0)
    parser.add_argument("--moon-mu", type=float, default=4.9048695e12)
    parser.add_argument("--initial-altitude", type=float, default=100_000.0)
    parser.add_argument(
        "--terminal-altitude",
        type=float,
        default=0.0,
        help="target altitude above the lunar sphere, metres",
    )
    parser.add_argument(
        "--terminal-beta-deg",
        type=float,
        default=None,
        help="optional fixed terminal angular distance, degrees",
    )
    return parser.parse_args()


def plot_solution(solution, output: Path) -> None:
    time_minutes = solution.time_s / 60.0
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(time_minutes, solution.state[2])
    axes[0, 0].set(ylabel="r / Rmoon", title="Radius")
    axes[0, 1].plot(time_minutes, solution.state[0], label="radial")
    axes[0, 1].plot(time_minutes, solution.state[1], label="tangential")
    axes[0, 1].set(title="Dimensionless velocity")
    axes[0, 1].legend()
    axes[1, 0].step(time_minutes, solution.throttle, where="post")
    axes[1, 0].set(xlabel="Time, min", ylabel="delta", title="Throttle")
    axes[1, 1].plot(time_minutes, solution.switching_function, label="S")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set(xlabel="Time, min", title="Switching function")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.example_500m:
        params = example_500m_parameters()
        target = example_500m_target()
        durations = [2800.0]
    else:
        params = PhysicalParameters(
            moon_radius_m=args.moon_radius,
            moon_mu_m3_s2=args.moon_mu,
            initial_mass_kg=args.initial_mass,
            dry_mass_kg=args.dry_mass,
            max_thrust_n=args.thrust,
            exhaust_velocity_m_s=args.exhaust_velocity,
            initial_altitude_m=args.initial_altitude,
        )
        target = LandingTarget(
            final_altitude_m=args.terminal_altitude,
            final_beta_rad=(
                None if args.terminal_beta_deg is None else args.terminal_beta_deg * 3.141592653589793 / 180.0
            ),
        )
        durations = list(range(600, 3601, 60)) if args.family else [args.duration]
    options = SolverOptions(nodes=args.nodes)
    solutions = solve_family(params, durations, options, target)
    for solution in solutions:
        directory = args.output / f"T_{solution.duration_s:g}_s"
        save_solution(solution, params, directory)
        plot_solution(solution, directory / "trajectory.png")
        print(
            f"T={solution.duration_s:7.1f} s  fuel={solution.objective_kg:9.3f} kg  "
            f"status={solution.status}"
        )
    save_family_summary(solutions, args.output / "summary.csv")


if __name__ == "__main__":
    main()
