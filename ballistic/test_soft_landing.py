import math

import numpy as np

from soft_landing import (
    LandingTarget,
    PhysicalParameters,
    dimensionless_rhs_numpy,
    make_scales,
)


def test_circular_orbit_is_radial_equilibrium() -> None:
    scales = make_scales(PhysicalParameters())
    state = np.array([0.0, math.sqrt(1.0 / scales.initial_radius), scales.initial_radius, 0.0, 0.0])
    derivative = dimensionless_rhs_numpy(state, np.zeros(2), 0.0, scales)
    assert abs(derivative[0]) < 1e-14
    assert derivative[2] == 0.0
    assert derivative[4] == 0.0


def test_full_throttle_fuel_rate_matches_rocket_model() -> None:
    params = PhysicalParameters()
    scales = make_scales(params)
    burned_dimensionless = scales.fuel_rate / scales.time_s
    expected = params.max_thrust_n / (params.exhaust_velocity_m_s * params.initial_mass_kg)
    assert math.isclose(burned_dimensionless, expected, rel_tol=1e-14)


def test_invalid_mass_is_rejected() -> None:
    params = PhysicalParameters(initial_mass_kg=5_000.0, dry_mass_kg=5_000.0)
    try:
        params.validate()
    except ValueError:
        return
    raise AssertionError("invalid masses were accepted")


def test_negative_terminal_altitude_is_rejected() -> None:
    try:
        LandingTarget(final_altitude_m=-1.0).validate()
    except ValueError:
        return
    raise AssertionError("negative terminal altitude was accepted")
