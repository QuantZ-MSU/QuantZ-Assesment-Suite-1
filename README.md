# QuantZ Assessment Set #1

## Instructions

1. **Time Management:** Allocate your time wisely. Each problem is challenging and may require substantial effort.
2. **Code Quality:** Write clean, efficient, and well-documented code. Use meaningful variable names and include comments where necessary to explain your logic.
3. **Accuracy:** Ensure your solutions are correct and handle edge cases appropriately.
4. **Testing:** Test your code with provided sample inputs to validate correctness.
5. **Submission:** Submit your solutions as a single Python file named `quantz_assessment.py` in the `src` folder.
6. **Test Cases:** We have implemented an Auto Grader, so only submit once you're super sure of your solution to avoid any issues.

## Scoring
Each problem has 10 test cases. Your score for each problem will be the number of test cases your solution passes out of 10.

## Problems

### Problem 1: Maximum Drawdown Calculation
**Question:**
Write a function `max_drawdown(prices)` that calculates the maximum drawdown of a given list of daily stock prices. Maximum drawdown is the maximum observed loss from a peak to a trough.


### Problem 2: Eigenportfolio Construction
**Question:**
Write a function `eigenportfolio(returns_matrix)` that takes a matrix of daily returns where each row represents a day and each column represents a stock. Calculate the eigenvector corresponding to the largest eigenvalue of the covariance matrix and use it to create portfolio weights.


### Problem 3: Monte Carlo Simulation for Option Pricing
**Question:**
Implement a Monte Carlo simulation to price a European call option using the Black-Scholes model. The function `monte_carlo_option_pricing(S, K, T, r, sigma, num_simulations)` should take the current stock price `S`, strike price `K`, time to maturity `T`, risk-free rate `r`, volatility `sigma`, and the number of simulations `num_simulations`.




### Problem 4: Value at Risk (VaR) Calculation
**Question:**
Implement a function `calculate_var(returns, confidence_level=0.95)` to calculate the 1-day 95% Value at Risk (VaR) using the historical simulation method.



### Problem 5: Principal Component Analysis (PCA) for Risk Management
**Question:**
Write a function `top_principal_components(returns_matrix, n_components=3)` to perform PCA on a given matrix of asset returns and identify the top 3 principal components.


### Problem 6: Solving Non-linear Optimization for Portfolio
**Question:**
Optimize a portfolio of assets to minimize the Value at Risk (VaR) at a 99% confidence level. The function `optimize_var(returns, confidence_level=0.99)` should use non-linear optimization techniques to return the optimal weights.



