# Multi-Armed Bandit Analysis

## Overview

This project explores various multi-armed bandit (MAB) algorithms through systematic experimentation across different problem settings. The goal is to understand how different algorithms handle the exploration-exploitation tradeoff under varying conditions: stationary vs. non-stationary environments, high variance scenarios, and sudden distribution shifts.

## Problem Formulation

The multi-armed bandit problem models sequential decision-making where an agent must repeatedly choose between k actions (arms), each providing stochastic rewards. The challenge lies in balancing:
- **Exploitation**: choosing actions that have yielded high rewards in the past
- **Exploration**: trying potentially suboptimal actions to discover better options

## Experimental Design

### Test Datasets

Four synthetic datasets were designed to evaluate algorithm performance under different conditions:

**Dataset A - Stationary Environment**
- 10 arms, 500 timesteps, 1000 samples
- True means drawn from N(0, 5), variances from U(0.8, 1.5)
- Tests basic exploration-exploitation balance

**Dataset B - Non-Stationary (Abrupt Change)**
- 4 arms, 1500 timesteps, 1000 samples
- Distribution switch at t=750 (50% mark)
- Tests adaptability to sudden environmental changes

**Dataset C - Non-Stationary (Gradual Drift)**
- 4 arms, 1500 timesteps, 1000 samples
- Means drift via cumulative Gaussian shocks (σ=0.25)
- Tests tracking slowly changing optima

**Dataset D - High Variance Stationary**
- 10 arms, 1000 timesteps, 1000 samples
- True variance = 5 (significantly higher than Dataset A)
- Tests robustness to noisy feedback

### Algorithms Evaluated

1. **ε-Greedy**: Random exploration with probability ε
2. **Optimistic Greedy**: Greedy with optimistic initialization
3. **Constant Step ε-Greedy**: ε-greedy with fixed learning rate (for non-stationary)
4. **UCB (Upper Confidence Bound)**: Systematic exploration via uncertainty estimates
5. **Gradient Bandit**: Preference-based softmax policy with baseline

## Key Observations

### Dataset A - Stationary Environment (Mean Score ≈ 0-7)

**Performance Rankings:**
1. **UCB (c=0.6)**: 6.866 - **Undeniable winner**
2. **UCB (c=1.5)**: 6.861 - Nearly identical performance
3. **Optimistic Greedy (qₛₜ=6)**: 6.841 - Extremely close third
4. **Optimistic Greedy (qₛₜ=7-12)**: 6.80-6.82 - Tight cluster
5. **Gradient Bandit (α=0.225)**: 6.545 - Solid but clearly behind
6. **ε-Greedy (ε=0.0625)**: 6.311 - Functional but outclassed
7. **Constant Step (best)**: 5.154 - **Terrible in stationary settings**

**Key Insight:** UCB dominates because it **systematically reduces uncertainty**. The top performers (UCB and Optimistic Greedy) are separated by only ~0.025 points, but there's a clear performance tier. Optimistic Greedy is competitive **IF** you guess qₛₜ near the true reward scale (≈6-8). Miss by much and you're in trouble—notice qₛₜ=0 only scores 4.270. Constant step-size is disastrous here because it throws away accumulated knowledge.

**Critical Trade-off:** UCB (6.866) vs Optimistic Greedy (6.841). If you have prior information about reward scales, Optimistic Greedy is simpler. If you don't, UCB wins by being self-tuning.

### Dataset B - Abrupt Distribution Shift at t=750 (Mean Score ≈ -0.5 to 2.5)

**Performance Rankings:**
1. **Constant Step (α=0.01, ε=0.0625)**: 2.234 - **DOMINATES by 3x margin!**
2. **Constant Step (α=0.05)**: 1.698
3. **Constant Step (α=0.1)**: 1.511
4. **UCB (c=6)**: 0.789 - Distant fourth, but best non-constant-step
5. **Everything else**: ≈0.05-0.19 (basically failing)
6. **High ε-Greedy, high α Constant Step**: Negative scores (worse than random)

**Critical Finding:** This is **not even close**. Constant step-size with small α (0.01-0.05) crushes all competitors. The smaller the step-size, the better—counterintuitively, you want **slow forgetting** not fast forgetting. Why? Because you need to average out noise while still adapting. Fast forgetting (α=0.625) gets you -0.183!

**The UCB Exception:** UCB with c=6 (aggressive exploration) manages 0.789—far behind constant-step but 4x better than standard algorithms. Large c forces re-exploration even after apparent convergence, providing some adaptation capability.

**The Disaster Zone:** Sample-average methods (standard ε-Greedy, standard UCB, Optimistic Greedy) score near zero or negative. They learned the wrong answer and can't unlearn it.

### Dataset C - Gradual Drift (Mean Score ≈ 4-8.5)

**Performance Rankings:**
1. **UCB (c=6)**: 8.453 - **Winner**
2. **UCB (c=1.5)**: 8.448 - Essentially tied
3. **Constant Step (α=0.1, ε=0.125)**: 7.818
4. **Optimistic Greedy (qₛₜ=10)**: 7.877
5. **Constant Step (α=0.1, ε=0.0625)**: 7.473
6. **ε-Greedy (ε=0.0625)**: 7.441
7. **Gradient Bandit (α=0.225)**: 5.252 - Notably poor

**Insight:** Gradual drift is **less punishing** than abrupt shifts. UCB's continuous exploration naturally tracks slow changes—it never fully "commits" so it naturally adapts. Constant step-size is only ~0.6 points behind, showing that forgetting helps but isn't critical when changes are slow.

**Surprising Result:** Standard ε-Greedy (7.441) nearly matches Constant Step (7.473)! With slow drift, even occasional random exploration is enough to notice the world changing. The gap between UCB and others (~0.5-1.0 points) is significant but not catastrophic.

### Dataset D - High Variance Stationary (Mean Score ≈ 9-13)

**Performance Rankings:**
1. **UCB (c=1.5)**: 12.937 - **Clear winner**
2. **UCB (c=6)**: 12.859
3. **Optimistic Greedy (qₛₜ=10)**: 12.721
4. **ε-Greedy (ε=0.0625)**: 12.100
5. **Gradient Bandit (α=0.225)**: 11.910
6. **Constant Step (α=0.1, ε=0.125)**: 11.183 - **Worst**

**Key Finding:** High variance **preserves the stationary hierarchy** but amplifies differences. UCB maintains its dominance, Constant Step remains worst. The performance gap (12.937 vs 11.183 = 1.75 points) is larger than in Dataset A, suggesting noise punishes poor algorithms more severely.

**Why Constant Step Fails:** With variance=5, each reward is very noisy. Sample-average (1/n) step sizes naturally implement "noise filtering" by weighting early samples less as n grows. Constant step-size (α=0.1) treats the 1000th noisy sample equally to the 1st, never achieving stable convergence. This is the flip side of its adaptation advantage—responsiveness to new data becomes vulnerability to noise.

## Comparative Analysis Across All Datasets

### The UCB Supremacy (Except Dataset B)

UCB wins or ties for first in **3 out of 4 datasets**:
- Dataset A (stationary): 6.866 vs 6.841 (Optimistic)
- Dataset C (gradual drift): 8.453 vs 7.818 (Constant Step)
- Dataset D (high variance): 12.937 vs 12.721 (Optimistic)

Only in Dataset B (abrupt shift) does UCB lose badly: 0.789 vs 2.234 (Constant Step = 3x better).

**Why UCB Dominates:** It naturally balances exploration and exploitation through uncertainty quantification. As uncertainty decreases, exploration naturally decreases—but never stops entirely.

### The Constant Step-Size Paradox

**The Trade-off:** Constant step-size implements exponential recency weighting—it "forgets" old data. This is:
- **Critical** when old data is wrong (non-stationary)
- **Harmful** when old data is valuable (stationary or high-variance)

### Optimistic Initialization: The High-Risk High-Reward Strategy

**Competitive Performance When Tuned:**
- Dataset A: 6.841 (only 0.025 behind UCB)
- Dataset D: 12.721 (only 0.216 behind UCB)

**Complete Failure When Mistuned:**
- Dataset A with qₛₜ=0: 4.270 (2.60 points behind UCB!)
- Dataset C with qₛₜ=0: 3.921 (4.53 points behind UCB!)

**The Lesson:** If you have domain knowledge about reward scales, Optimistic Greedy is simple and effective. Without it, you're gambling. UCB eliminates this guesswork.

### Gradient Bandit: The Consistent Underperformer

Gradient Bandit ranks 5th or 6th in all datasets:
- Dataset A: 6.545 (behind UCB, Optimistic, even ε-Greedy)
- Dataset B: 0.161 (adapt slowly, though "on upward trajectory")
- Dataset C: 5.252 (**worst** performer)
- Dataset D: 11.910 (middle of the pack)

**Why It Struggles:** Gradient bandit updates *preferences*, not value estimates. This indirect approach is:
- Less sample-efficient than direct Q-value learning
- Sensitive to step-size tuning
- Vulnerable when the baseline lags behind true rewards

**When It Might Help:** Very large action spaces where softmax policies are natural. Not demonstrated in this 4-10 arm setting.

## Cross-Dataset Insights

### The Stationarity Assumption is Critical

### Variance Amplifies Existing Weaknesses

Dataset D (high variance) doesn't change the algorithm rankings, it just **makes differences larger**:

## Final Thoughts

This analysis demonstrates that **context matters more than algorithmic sophistication**. The "best" algorithm depends entirely on whether your environment is stationary:

- **Stationary world:** UCB's principled exploration dominates
- **Changing world:** Constant step-size's "forgetting" is essential

There is no one-size-fits-all solution to the exploration-exploitation dilemma. Understanding your problem structure—particularly stationarity and reward variance—is more valuable than memorizing algorithm rankings.

## Future Directions

- Thompson Sampling for Bayesian exploration
- Contextual bandits for state-dependent rewards
- Adversarial bandits for worst-case robustness
- Regret bounds analysis and comparison to theoretical limits

---

*This analysis demonstrates the practical application of reinforcement learning principles to sequential decision-making under uncertainty, highlighting the importance of matching algorithmic assumptions to problem characteristics.*
*While the code file was created manually, LLM models were used to draft this report so there might be some inaccuracies.*
