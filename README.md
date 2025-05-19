## Exam preparation

### Part I
#### Component 1: Direct Sampling/MCMC Introduction
* Rejection sampling (3 problems $\rightarrow$ 3.5h) 🗹
* Inversion method (1 problem $\rightarrow$ 0.5h) 🗹
* Conditional distributions for sampling by means of MCMC (1 problem)

#### Component 2: Monte Carlo Methods
* Slice sampler, Gibbs sampler (1 problem)
* Metropolis–Hastings algorithm / Markov transition density (3 problems)
* SNIS (1 problem)

### Part II
#### Component 3: Bayesian Inference
* Bayesian posterior sampling with MCMC (5 problems)

#### Component 4: Expectation-Maximization (EM)
* EM algorithm for missing data (5 problems)

### Part III
#### Component 5: Specialized Monte Carlo and Estimators
* Particle filters (SMC) (1 problem)
* Hamiltonian Monte Carlo (1 problem)
* Generalized rejection sampling (1 problem)
* Resampling methods (1 problem)
* Unbiased estimators under transformations (1 problem) 

### Miscellaneous
* Differentiation and Integration Practice

## Monte Carlo in general
### MCMC Methods
1. Gibbs Sampling
2. Metropolis-Hastings Algorithm
3. Slice Sampling
4. Hamiltonian Monte Carlo (HMC) - MCMC with Hamiltonian dynamics
5. Langevin Monte Carlo
6. Metropolis-within-Gibbs (Hybrid MCMC)

### Alternatives to MCMC
1. Sequential Monte Carlo (SMC)
2. Self-Normalized Importance Sampling (SNIS)
3. Sequential Importance Sampling with Resampling (SISR)



------------------------------

## Notes on "5 Markov Chain Monte Carlo (MCMC) Methods"
### Gibbs sampler
* updates one variable from the variable vector at a time given its conditional distribution (how it behaves given the other variables)
* works becasue each step uses conditional distributions, which are pieces of the full distribution
* the samples match the target distribution after a while (called "burn-in") 
* apparently it allows the target distribution to be a stationary distribution (need to prove it)
* pros:
    * Easy when conditional distributions are simple (like in the script’s examples, where they’re Gamma or Normal).
* cons:
    * Slow if variables are strongly related (e.g., in the regression example on page 127, where $\theta_1$ and $\theta_2$ are correlated, causing slower mixing) 

### Slice sampler
* special case of the Gibbs sampler
* it is a trick to sample from a distribution when it’s hard to do directly
* 