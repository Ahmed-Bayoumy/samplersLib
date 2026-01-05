import numpy as np
import pytest

# Import the class under test
from samplersLib.particle import Particle


@pytest.fixture
def bounds():
    """2‑D search space: each dimension in [‑5, 5]"""
    return np.array([[-5, -5], [5, 5]])


def test_instantiation_random(bounds):
    """Particle should create a random position/velocity inside the bounds."""
    p = Particle(bounds)

    # position length matches dimensionality
    assert p.position.shape == (bounds.shape[1],)

    # each coordinate lies inside the bounds
    assert np.all(p.position >= bounds[0])
    assert np.all(p.position <= bounds[1])

    # velocity is in [‑1, 1] for each dimension
    assert np.all(p.velocity >= -1) and np.all(p.velocity <= 1)

    # best_position/value are initialised correctly
    np.testing.assert_array_equal(p.best_position, p.position)
    assert p.best_value == float("-inf")


def test_instantiation_with_position(bounds):
    """Providing an explicit position should be respected."""
    pos = np.array([1.0, -2.0])
    p = Particle(bounds, pos=pos)

    np.testing.assert_array_equal(p.position, pos)
    np.testing.assert_array_equal(p.best_position, pos)
    assert p.best_value == float("-inf")


def test_update_velocity(bounds):
    """Velocity update must follow the PSO equation."""
    np.random.seed(0)          # deterministic random numbers for the test
    p = Particle(bounds)

    # Freeze current velocity to a known value
    p.velocity = np.array([0.5, -0.5])

    # Define a global best that is different from the particle's current position
    global_best = np.array([2.0, 2.0])

    # Use known weights
    inertia = 0.4
    cognitive = 0.3
    social = 0.2

    # Capture the random factors r1, r2 generated inside the method
    # With the seed above, np.random.rand(2) returns (0.5488135, 0.71518937)
    r1, r2 = 0.5488135, 0.71518937

    # Expected velocity according to the formula
    expected = (inertia * p.velocity +
                cognitive * r1 * (p.best_position - p.position) +
                social * r2 * (global_best - p.position))

    p.update_velocity(global_best, inertia, cognitive, social)

    np.testing.assert_allclose(p.velocity, expected, rtol=0.055)


def test_update_position_respects_bounds(bounds):
    """After moving, the particle must stay inside the bounds."""
    p = Particle(bounds)

    # Force a velocity that would push the particle outside the bounds
    p.velocity = np.array([10.0, -10.0])

    # Store original position for later comparison
    original = p.position.copy()

    p.update_position(bounds)

    # Position should have moved, then been clipped
    assert not np.allclose(p.position, original)

    # Every coordinate must be within the bounds
    assert np.all(p.position >= bounds[0])
    assert np.all(p.position <= bounds[1])


def test_evaluate_updates_best(bounds):
    """evaluate should update best_value/best_position only when improvement occurs."""
    p = Particle(bounds)

    # Simple quadratic target: higher value = closer to origin
    def target(x):
        return -np.sum(x ** 2)

    # First evaluation – should become the best
    p.evaluate(target)
    first_best_val = p.best_value
    first_best_pos = p.best_position.copy()

    # Move the particle to a worse location manually
    p.position = np.array([10.0, 10.0])
    p.evaluate(target)

    # Best should stay unchanged because the new value is lower
    assert p.best_value == first_best_val
    np.testing.assert_array_equal(p.best_position, first_best_pos)

    # Move to a better location
    p.position = np.array([0.0, 0.0])
    p.evaluate(target)

    # Now best should be updated
    assert p.best_value > first_best_val
    np.testing.assert_array_equal(p.best_position, p.position)


# Optional: run a quick integration sanity‑check (not required for coverage)
def test_particle_full_cycle(bounds):
    """Run a few PSO steps and ensure no exceptions and reasonable behaviour."""
    np.random.seed(42)
    p = Particle(bounds)

    def target(x):
        # multimodal toy function
        return np.sin(np.linalg.norm(x))

    for _ in range(5):
        p.update_velocity(p.position, 0.5, 0.5, 0.5)  # using its own position as global best
        p.update_position(bounds)
        p.evaluate(target)

    # After a few steps the best value should be finite and within expected range
    assert np.isfinite(p.best_value)
    assert -1.0 <= p.best_value <= 1.0