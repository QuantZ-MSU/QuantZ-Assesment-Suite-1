import numpy as np
from src.quantz_assessment import max_drawdown, eigenportfolio, monte_carlo_option_pricing, calculate_var, top_principal_components, optimize_var

def test_max_drawdown():
    test_cases = [
        ([100, 120, 130, 90, 85, 95, 80, 120, 150], -0.4),
        ([100, 90, 80, 70, 60, 50], -0.5),
        ([100, 105, 110, 115, 120], 0.0),
        ([100, 110, 90, 110, 90, 120], -0.18181818181818182),
        ([150, 140, 130, 120, 110, 100, 110, 120], -0.3333333333333333),
        ([100], 0.0),
        ([120, 110, 100, 80, 60, 40], -0.6666666666666666),
        ([100, 100, 100, 100], 0.0),
        ([50, 100, 150, 200, 150, 100, 50], -0.75),
        ([100, 90, 80, 70, 60, 70, 80, 90, 100], -0.4)
    ]
    passed = 0
    for prices, expected in test_cases:
        result = max_drawdown(prices)
        assert np.isclose(result, expected), f"Failed for {prices}"
        passed += 1
    print(f"Passed {passed}/{len(test_cases)} test cases for max_drawdown")

def test_eigenportfolio():
    test_cases = [
        (np.random.randn(100, 5), "check_normalization"),
        (np.random.randn(200, 10), "check_normalization"),
        (np.ones((50, 5)), np.array([0.2] * 5)),
        (np.vstack([np.zeros((99, 5)), np.ones(5)]), "check_normalization"),
        (np.eye(10), "check_normalization"),
        (np.diag([1, 2, 3, 4, 5]), "check_normalization"),
        (np.random.randn(1000, 3), "check_normalization"),
        (np.random.randn(365, 7), "check_normalization"),
        (np.random.randn(52, 4), "check_normalization"),
        (np.random.randn(250, 6), "check_normalization")
    ]
    def is_normalized(weights):
        return np.isclose(np.sum(weights), 1.0) and all(weights >= 0)
    passed = 0
    for returns_matrix, expected in test_cases:
        weights = eigenportfolio(returns_matrix)
        if expected == "check_normalization":
            assert is_normalized(weights), f"Weights not normalized for {returns_matrix}"
        elif np.allclose(weights, expected):
            passed += 1
        print(f"Passed {passed}/{len(test_cases)} test cases for eigenportfolio")

def test_monte_carlo_option_pricing():
    test_cases = [
        (100, 105, 1, 0.05, 0.2, 100000, 8.0),
        (100, 110, 1, 0.05, 0.2, 100000, 5.5),
        (100, 100, 1, 0.05, 0.2, 100000, 10.5),
        (50, 60, 1, 0.05, 0.2, 100000, 1.8),
        (100, 150, 1, 0.05, 0.2, 100000, 0.1),
        (100, 105, 0.5, 0.05, 0.2, 100000, 5.4),
        (100, 105, 2, 0.05, 0.2, 100000, 15.6),
        (100, 105, 1, 0.1, 0.2, 100000, 11.0),
        (100, 105, 1, 0.05, 0.1, 100000, 4.3),
        (200, 210, 1, 0.05, 0.2, 100000, 19.2)
    ]
    passed = 0
    for S, K, T, r, sigma, num_simulations, expected in test_cases:
        result = monte_carlo_option_pricing(S, K, T, r, sigma, num_simulations)
        assert np.isclose(result, expected, atol=1.0), f"Failed for {S}, {K}, {T}, {r}, {sigma}, {num_simulations}"
        passed += 1
    print(f"Passed {passed}/{len(test_cases)} test cases for monte_carlo_option_pricing")

def test_calculate_var():
    test_cases = [
        (np.random.normal(0, 0.01, 1000), 0.95, -0.016),
        (np.random.normal(0, 0.02, 1000), 0.95, -0.032),
        (np.random.normal(0, 0.01, 10000), 0.99, -0.027),
        (np.random.normal(0, 0.01, 500), 0.95, -0.017),
        (np.random.normal(0, 0.015, 1000), 0.95, -0.024),
        (np.random.normal(0, 0.01, 1000), 0.90, -0.012),
        (np.random.normal(0, 0.02, 1000), 0.99, -0.044),
        (np.random.normal(0, 0.005, 1000), 0.95, -0.008),
        (np.random.normal(0, 0.01, 1000), 0.99, -0.021),
        (np.random.normal(0, 0.02, 10000), 0.95, -0.032)
    ]
    passed = 0
    for returns, confidence_level, expected in test_cases:
        result = calculate_var(returns, confidence_level)
        assert np.isclose(result, expected, atol=0.005), f"Failed for {returns}, {confidence_level}"
        passed += 1
    print(f"Passed {passed}/{len(test_cases)} test cases for calculate_var")

def test_top_principal_components():
    test_cases = [
        (np.random.randn(100, 5), 3),
        (np.random.randn(200, 10), 3),
        (np.ones((50, 5)), 3),
        (np.vstack([np.zeros((99, 5)), np.ones(5)]), 3),
        (np.eye(10), 3),
        (np.diag([1, 2, 3, 4, 5]), 3),
        (np.random.randn(1000, 3), 3),
        (np.random.randn(365, 7), 3),
        (np.random.randn(52, 4), 3),
        (np.random.randn(250, 6), 3)
    ]
    passed = 0
    for returns_matrix, n_components in test_cases:
        components = top_principal_components(returns_matrix, n_components)
        assert components.shape == (n_components, returns_matrix.shape[1]), f"Failed for {returns_matrix}, {n_components}"
        passed += 1
    print(f"Passed {passed}/{len(test_cases)} test cases for top_principal_components")

def test_optimize_var():
    test_cases = [
        (np.random.randn(1000, 5), 0.99),
        (np.random.randn(500, 10), 0.99),
        (np.random.randn(1000, 3), 0.99),
        (np.random.randn(365, 7), 0.99),
        (np.random.randn(52, 4), 0.99),
        (np.random.randn(250, 6), 0.99),
        (np.random.randn(1000, 5), 0.95),
        (np.random.randn(500, 10), 0.95),
        (np.random.randn(1000, 3), 0.95),
        (np.random.randn(365, 7), 0.95)
    ]
    passed = 0
    for returns_matrix, confidence_level in test_cases:
        weights = optimize_var(returns_matrix, confidence_level)
        assert np.isclose(np.sum(weights), 1.0) and all(weights >= 0), f"Failed for {returns_matrix}, {confidence_level}"
        passed += 1
    print(f"Passed {passed}/{len(test_cases)} test cases for optimize_var")

if __name__ == "__main__":
    test_max_drawdown()
    test_eigenportfolio()
    test_monte_carlo_option_pricing()
    test_calculate_var()
    test_top_principal_components()
    test_optimize_var()
    total_score = sum([test_max_drawdown(), test_eigenportfolio(), test_monte_carlo_option_pricing(), test_calculate_var(), test_top_principal_components(), test_optimize_var()])
    print(f"Total Score: {total_score}/60")
