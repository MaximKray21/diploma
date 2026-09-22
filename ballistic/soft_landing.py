"""Numerical solution of the planar minimum-fuel lunar landing problem.

The equations follow the model in ``Теоретические основы оптимизации мягкой
посадки- v2.pdf``.  A direct multiple-shooting transcription is used because it
is considerably less sensitive to an initial costate guess than plain indirect
shooting.  All equations solved by IPOPT are nondimensional.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class PhysicalParameters:
    moon_radius_m: float = 1_737_400.0
    moon_mu_m3_s2: float = 4.9048695e12
    initial_mass_kg: float = 15_000.0
    dry_mass_kg: float = 5_000.0
    max_thrust_n: float = 60_000.0
    exhaust_velocity_m_s: float = 3_050.0
    initial_altitude_m: float = 100_000.0

    def validate(self) -> None:
        positive = {
            "moon_radius_m": self.moon_radius_m,
            "moon_mu_m3_s2": self.moon_mu_m3_s2,
            "initial_mass_kg": self.initial_mass_kg,
            "dry_mass_kg": self.dry_mass_kg,
            "max_thrust_n": self.max_thrust_n,
            "exhaust_velocity_m_s": self.exhaust_velocity_m_s,
        }
        bad = [name for name, value in positive.items() if value <= 0]
        if bad:
            raise ValueError(f"Parameters must be positive: {', '.join(bad)}")
        if self.dry_mass_kg >= self.initial_mass_kg:
            raise ValueError("dry_mass_kg must be less than initial_mass_kg")
        if self.initial_altitude_m <= 0:
            raise ValueError("initial_altitude_m must be positive")


@dataclass(frozen=True)
class SolverOptions:
    nodes: int = 80
    tolerance: float = 1e-8
    max_iterations: int = 2_000
    validation_points_per_interval: int = 5
    ipopt_print_level: int = 0

    def validate(self) -> None:
        if self.nodes < 10:
            raise ValueError("nodes must be at least 10")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if self.validation_points_per_interval < 2:
            raise ValueError("validation_points_per_interval must be at least 2")


@dataclass(frozen=True)
class LandingTarget:
    """Terminal conditions for a fixed-duration powered descent.

    The terminal angle is intentionally optional.  Omitting it preserves the
    free landing-site formulation of the theoretical document.
    """

    final_altitude_m: float = 0.0
    final_beta_rad: float | None = None

    def validate(self) -> None:
        if self.final_altitude_m < 0:
            raise ValueError("final_altitude_m cannot be negative")
        if self.final_beta_rad is not None and not math.isfinite(self.final_beta_rad):
            raise ValueError("final_beta_rad must be finite when specified")


@dataclass
class Scales:
    length_m: float
    time_s: float
    velocity_m_s: float
    thrust_acceleration: float
    fuel_rate: float
    initial_radius: float


@dataclass
class LandingSolution:
    duration_s: float
    target: LandingTarget
    time_s: np.ndarray
    state: np.ndarray
    thrust_vector: np.ndarray
    throttle: np.ndarray
    costate: np.ndarray
    switching_function: np.ndarray
    hamiltonian: np.ndarray
    status: str
    objective_kg: float
    diagnostics: dict[str, float]

    def warm_start(self) -> dict[str, np.ndarray]:
        return {
            "state": self.state.copy(),
            "thrust_vector": self.thrust_vector[:, :-1].copy(),
            "throttle": self.throttle[:-1].copy(),
        }


def make_scales(params: PhysicalParameters) -> Scales:
    params.validate()
    length = params.moon_radius_m
    time = math.sqrt(length**3 / params.moon_mu_m3_s2)
    velocity = length / time
    gravity = params.moon_mu_m3_s2 / length**2
    return Scales(
        length_m=length,
        time_s=time,
        velocity_m_s=velocity,
        thrust_acceleration=(params.max_thrust_n / params.initial_mass_kg) / gravity,
        fuel_rate=(params.max_thrust_n * time)
        / (params.exhaust_velocity_m_s * params.initial_mass_kg),
        initial_radius=1.0 + params.initial_altitude_m / length,
    )


def example_500m_parameters() -> PhysicalParameters:
    """Physical values from ``Примеры решений задачи посадки на Луну.pdf``.

    The source does not state a dry mass.  5,000 kg is an inactive resource
    bound for the published 5,554.888 kg state at 500 m and does not alter the
    unconstrained minimum-fuel solution.
    """

    standard_gravity = 9.80665
    return PhysicalParameters(
        moon_radius_m=1_737_530.0,
        moon_mu_m3_s2=4.902798e12,
        initial_mass_kg=10_000.0,
        dry_mass_kg=5_000.0,
        max_thrust_n=2_500.0 * standard_gravity,
        exhaust_velocity_m_s=320.0 * standard_gravity,
        initial_altitude_m=122_740.3,
    )


def example_500m_target() -> LandingTarget:
    """Terminal point of the detailed descent table in the example PDF."""

    return LandingTarget(final_altitude_m=500.0, final_beta_rad=math.radians(132.7608))


def dimensionless_rhs_numpy(
    state: np.ndarray, thrust_vector: np.ndarray, throttle: float, scales: Scales
) -> np.ndarray:
    radial_velocity, tangential_velocity, radius, _beta, fuel_fraction = state
    mass_fraction = 1.0 - fuel_fraction
    radial_thrust, tangential_thrust = thrust_vector
    return np.array(
        [
            -1.0 / radius**2
            + tangential_velocity**2 / radius
            + scales.thrust_acceleration * radial_thrust / mass_fraction,
            -radial_velocity * tangential_velocity / radius
            + scales.thrust_acceleration * tangential_thrust / mass_fraction,
            radial_velocity,
            tangential_velocity / radius,
            scales.fuel_rate * throttle,
        ]
    )


def _casadi_rhs(state: ca.MX, thrust: ca.MX, throttle: ca.MX, scales: Scales) -> ca.MX:
    radial_velocity, tangential_velocity, radius, _, fuel_fraction = ca.vertsplit(state)
    mass_fraction = 1.0 - fuel_fraction
    return ca.vertcat(
        -1.0 / radius**2
        + tangential_velocity**2 / radius
        + scales.thrust_acceleration * thrust[0] / mass_fraction,
        -radial_velocity * tangential_velocity / radius
        + scales.thrust_acceleration * thrust[1] / mass_fraction,
        radial_velocity,
        tangential_velocity / radius,
        scales.fuel_rate * throttle,
    )


def _initial_guess(
    scales: Scales, final_time: float, nodes: int, target: LandingTarget
) -> dict[str, np.ndarray]:
    fraction = np.linspace(0.0, 1.0, nodes + 1)
    state = np.zeros((5, nodes + 1))
    state[1] = math.sqrt(1.0 / scales.initial_radius) * (1.0 - fraction)
    target_radius = 1.0 + target.final_altitude_m / scales.length_m
    state[2] = scales.initial_radius + (target_radius - scales.initial_radius) * fraction
    terminal_beta = (
        target.final_beta_rad
        if target.final_beta_rad is not None
        else 0.45 * final_time * math.sqrt(1.0 / scales.initial_radius)
    )
    state[3] = terminal_beta * fraction
    state[4] = 0.45 * fraction
    thrust = np.empty((2, nodes))
    thrust[0] = 0.40
    thrust[1] = -0.75
    return {"state": state, "thrust_vector": thrust, "throttle": np.full(nodes, 0.88)}


def solve_landing(
    params: PhysicalParameters,
    duration_s: float,
    options: SolverOptions = SolverOptions(),
    warm_start: dict[str, np.ndarray] | None = None,
    target: LandingTarget = LandingTarget(),
) -> LandingSolution:
    """Solve one fixed-duration landing problem.

    The thrust-vector components are ``delta*sin(theta)`` and
    ``delta*cos(theta)``.  The second-order-cone constraint on this vector is
    equivalent to ``0 <= delta <= 1``; minimizing fuel makes the inequality
    tight except at numerical tolerance.
    """

    params.validate()
    options.validate()
    target.validate()
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")

    scales = make_scales(params)
    nodes = options.nodes
    final_time = duration_s / scales.time_s
    step_size = final_time / nodes
    fuel_limit = 1.0 - params.dry_mass_kg / params.initial_mass_kg
    target_radius = 1.0 + target.final_altitude_m / scales.length_m

    opti = ca.Opti()
    state = opti.variable(5, nodes + 1)
    thrust = opti.variable(2, nodes)
    throttle = opti.variable(1, nodes)
    dynamic_constraints: list[ca.MX] = []

    initial_state = ca.vertcat(
        0.0, math.sqrt(1.0 / scales.initial_radius), scales.initial_radius, 0.0, 0.0
    )
    opti.subject_to(state[:, 0] == initial_state)

    def rk4(current: ca.MX, control: ca.MX, level: ca.MX) -> ca.MX:
        k1 = _casadi_rhs(current, control, level, scales)
        k2 = _casadi_rhs(current + 0.5 * step_size * k1, control, level, scales)
        k3 = _casadi_rhs(current + 0.5 * step_size * k2, control, level, scales)
        k4 = _casadi_rhs(current + step_size * k3, control, level, scales)
        return current + step_size * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    for index in range(nodes):
        equality = state[:, index + 1] == rk4(
            state[:, index], thrust[:, index], throttle[0, index]
        )
        dynamic_constraints.append(equality)
        opti.subject_to(equality)

    opti.subject_to(state[0:3, -1] == ca.vertcat(0.0, 0.0, target_radius))
    if target.final_beta_rad is not None:
        opti.subject_to(state[3, -1] == target.final_beta_rad)
    opti.subject_to(ca.sum1(thrust**2) <= throttle**2)
    opti.subject_to(opti.bounded(0.0, throttle, 1.0))
    opti.subject_to(state[2, :] >= 1.0)
    opti.subject_to(opti.bounded(0.0, state[4, :], fuel_limit))
    opti.minimize(state[4, -1])

    guess = warm_start or _initial_guess(scales, final_time, nodes, target)
    if guess["state"].shape != (5, nodes + 1):
        raise ValueError("warm-start state has an incompatible shape")
    opti.set_initial(state, guess["state"])
    opti.set_initial(thrust, guess["thrust_vector"])
    opti.set_initial(throttle, guess["throttle"])

    opti.solver(
        "ipopt",
        {"expand": True, "print_time": False},
        {
            "max_iter": options.max_iterations,
            "tol": options.tolerance,
            "constr_viol_tol": max(options.tolerance * 0.1, 1e-10),
            "acceptable_tol": max(10.0 * options.tolerance, 1e-7),
            "acceptable_iter": 10,
            "print_level": options.ipopt_print_level,
            "sb": "yes",
        },
    )
    try:
        result = opti.solve()
    except RuntimeError as error:
        status = opti.stats().get("return_status", "unknown")
        raise RuntimeError(f"IPOPT failed for T={duration_s:g} s: {status}") from error

    state_value = np.asarray(result.value(state), dtype=float)
    thrust_intervals = np.asarray(result.value(thrust), dtype=float)
    throttle_intervals = np.asarray(result.value(throttle), dtype=float).reshape(nodes)
    terminal_costates = np.column_stack(
        [np.asarray(result.value(opti.dual(item)), dtype=float).reshape(5) for item in dynamic_constraints]
    )
    costate = np.empty((5, nodes + 1))
    costate[:, 1:] = terminal_costates
    costate[:, 0] = costate[:, 1]
    # NLP uses terminal cost q(T); convert its adjoint to the integral-cost
    # convention used in the document: p_q = p_q_terminal + 1.
    costate[4] += 1.0

    thrust_value = np.column_stack((thrust_intervals, thrust_intervals[:, -1]))
    throttle_value = np.append(throttle_intervals, throttle_intervals[-1])
    switching, hamiltonian = _pontryagin_diagnostics(
        state_value, thrust_value, throttle_value, costate, scales
    )
    time_s = np.linspace(0.0, duration_s, nodes + 1)
    diagnostics = _validate_solution(
        params, scales, duration_s, state_value, thrust_intervals, throttle_intervals, options, target
    )
    diagnostics["hamiltonian_range"] = float(np.ptp(hamiltonian))
    diagnostics["switch_control_mismatch"] = float(
        np.mean((switching[:-1] > 0.0) != (throttle_intervals > 0.5))
    )
    active = throttle_intervals > 1e-2
    if np.any(active):
        costate_velocity = costate[:2, :-1][:, active]
        control_direction = thrust_intervals[:, active]
        direction_cosine = np.sum(costate_velocity * control_direction, axis=0) / (
            np.linalg.norm(costate_velocity, axis=0)
            * np.linalg.norm(control_direction, axis=0)
        )
        diagnostics["minimum_thrust_costate_direction_cosine"] = float(
            np.min(direction_cosine)
        )

    return LandingSolution(
        duration_s=duration_s,
        target=target,
        time_s=time_s,
        state=state_value,
        thrust_vector=thrust_value,
        throttle=throttle_value,
        costate=costate,
        switching_function=switching,
        hamiltonian=hamiltonian,
        status=str(opti.stats().get("return_status", "Solve_Succeeded")),
        objective_kg=float(state_value[4, -1] * params.initial_mass_kg),
        diagnostics=diagnostics,
    )


def _pontryagin_diagnostics(
    state: np.ndarray,
    thrust: np.ndarray,
    throttle: np.ndarray,
    costate: np.ndarray,
    scales: Scales,
) -> tuple[np.ndarray, np.ndarray]:
    radial_velocity, tangential_velocity, radius, _, fuel = state
    p_radial_velocity, p_tangential_velocity, p_radius, _, p_fuel = costate
    mass = 1.0 - fuel
    costate_velocity_norm = np.hypot(p_radial_velocity, p_tangential_velocity)
    switching = (
        scales.thrust_acceleration * costate_velocity_norm / mass
        + scales.fuel_rate * (p_fuel - 1.0)
    )
    ballistic = (
        p_radial_velocity * (-1.0 / radius**2 + tangential_velocity**2 / radius)
        + p_tangential_velocity * (-radial_velocity * tangential_velocity / radius)
        + p_radius * radial_velocity
    )
    powered = scales.thrust_acceleration * (
        p_radial_velocity * thrust[0] + p_tangential_velocity * thrust[1]
    ) / mass + scales.fuel_rate * (p_fuel - 1.0) * throttle
    return switching, ballistic + powered


def _validate_solution(
    params: PhysicalParameters,
    scales: Scales,
    duration_s: float,
    state: np.ndarray,
    thrust: np.ndarray,
    throttle: np.ndarray,
    options: SolverOptions,
    target: LandingTarget,
) -> dict[str, float]:
    nodes = throttle.size
    interval = duration_s / scales.time_s / nodes
    current = state[:, 0].copy()
    minimum_radius = current[2]
    for index in range(nodes):
        sample_times = np.linspace(0.0, interval, options.validation_points_per_interval)
        integration = solve_ivp(
            lambda _, value: dimensionless_rhs_numpy(value, thrust[:, index], throttle[index], scales),
            (0.0, interval),
            current,
            t_eval=sample_times,
            rtol=min(options.tolerance, 1e-10),
            atol=min(options.tolerance * 0.1, 1e-12),
        )
        if not integration.success:
            raise RuntimeError(f"Independent validation integration failed: {integration.message}")
        minimum_radius = min(minimum_radius, float(np.min(integration.y[2])))
        current = integration.y[:, -1]

    target_radius = 1.0 + target.final_altitude_m / scales.length_m
    terminal_error = current[:3] - np.array([0.0, 0.0, target_radius])
    vector_norm = np.linalg.norm(thrust, axis=0)
    fuel_by_quadrature = (
        np.sum(throttle) * duration_s / nodes * params.max_thrust_n / params.exhaust_velocity_m_s
    )
    diagnostics = {
        "terminal_radial_velocity_m_s": float(terminal_error[0] * scales.velocity_m_s),
        "terminal_tangential_velocity_m_s": float(terminal_error[1] * scales.velocity_m_s),
        "terminal_altitude_error_m": float(terminal_error[2] * scales.length_m),
        "final_altitude_m": float((current[2] - 1.0) * scales.length_m),
        "minimum_altitude_m": float((minimum_radius - 1.0) * scales.length_m),
        "fuel_balance_error_kg": float(
            state[4, -1] * params.initial_mass_kg - fuel_by_quadrature
        ),
        "maximum_thrust_constraint_violation": float(np.max(vector_norm - throttle)),
        "remaining_fuel_kg": float(
            params.initial_mass_kg - params.dry_mass_kg - state[4, -1] * params.initial_mass_kg
        ),
        "final_mass_kg": float(params.initial_mass_kg * (1.0 - state[4, -1])),
    }
    if target.final_beta_rad is not None:
        diagnostics["terminal_beta_error_deg"] = float(
            math.degrees(current[3] - target.final_beta_rad)
        )
    return diagnostics


def solve_family(
    params: PhysicalParameters,
    durations_s: list[float],
    options: SolverOptions = SolverOptions(),
    target: LandingTarget = LandingTarget(),
) -> list[LandingSolution]:
    solutions: list[LandingSolution] = []
    warm_start = None
    for duration in durations_s:
        try:
            solution = solve_landing(params, duration, options, warm_start, target)
        except RuntimeError:
            if warm_start is None:
                raise
            # A continuation guess can occasionally lead IPOPT to the wrong
            # local branch.  Retry once from the generic physical guess.
            solution = solve_landing(params, duration, options, None, target)
        solutions.append(solution)
        warm_start = solution.warm_start()
    return solutions


def save_solution(solution: LandingSolution, params: PhysicalParameters, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    mass = params.initial_mass_kg * (1.0 - solution.state[4])
    angle = np.full_like(solution.throttle, np.nan)
    active = solution.throttle > 1e-7
    angle[active] = np.arctan2(
        solution.thrust_vector[0, active], solution.thrust_vector[1, active]
    )
    table = np.column_stack(
        (
            solution.time_s,
            solution.state[2] * params.moon_radius_m,
            (solution.state[2] - 1.0) * params.moon_radius_m,
            solution.state[0] * math.sqrt(params.moon_mu_m3_s2 / params.moon_radius_m),
            solution.state[1] * math.sqrt(params.moon_mu_m3_s2 / params.moon_radius_m),
            solution.state[3],
            solution.state[4] * params.initial_mass_kg,
            mass,
            solution.throttle,
            angle,
            solution.thrust_vector[0],
            solution.thrust_vector[1],
            *solution.costate,
            solution.switching_function,
            solution.hamiltonian,
        )
    )
    header = ",".join(
        [
            "time_s", "radius_m", "altitude_m", "radial_velocity_m_s",
            "tangential_velocity_m_s", "beta_rad", "fuel_used_kg", "mass_kg",
            "throttle", "theta_rad", "thrust_radial_fraction", "thrust_tangential_fraction",
            "p_vr", "p_vt", "p_r", "p_beta", "p_fuel", "switching_function",
            "hamiltonian",
        ]
    )
    np.savetxt(directory / "trajectory.csv", table, delimiter=",", header=header, comments="")
    metadata: dict[str, Any] = {
        "duration_s": solution.duration_s,
        "target": asdict(solution.target),
        "status": solution.status,
        "fuel_used_kg": solution.objective_kg,
        "parameters": asdict(params),
        "diagnostics": solution.diagnostics,
    }
    (directory / "report.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_family_summary(solutions: list[LandingSolution], path: Path) -> None:
    rows = np.array(
        [
            [
                item.duration_s,
                item.objective_kg,
                item.diagnostics["remaining_fuel_kg"],
                item.diagnostics["terminal_radial_velocity_m_s"],
                item.diagnostics["terminal_tangential_velocity_m_s"],
                item.diagnostics["terminal_altitude_error_m"],
                item.diagnostics["minimum_altitude_m"],
            ]
            for item in solutions
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        rows,
        delimiter=",",
        comments="",
        header=(
            "duration_s,fuel_used_kg,remaining_fuel_kg,terminal_radial_velocity_m_s,"
            "terminal_tangential_velocity_m_s,terminal_altitude_error_m,minimum_altitude_m"
        ),
    )
