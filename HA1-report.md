## HA1 Report: A hidden Markov model for mobility tracking

### Problem 1: Motion model

#### Is $\{X_n\}_{n \in \mathcal{N}}$ a Markov chain?

No. The definition of a Markov chain requires that the probability distribution of the next state of kinematics-vector $X_{n+1}$ depends only on the current state $X_n$. 

In the given model, the next state $X_{n+1}$ depends not only on the current state $X_n$, but also on the current driving command $Z_n$, where $\{Z_n\}_{n \in \mathcal{N}}$ is an independent Markov chain. Thus knowing $X_n$ alone is insufficient to determine the distribution of $X_{n+1}$, which means $\{X_n\}_{n \in \mathcal{N}}$ does not satisfy the Markov property. 

The noise $W_{n+1}$ is mutually independent completely stochastic component determined at time-step $n+1$. This means we can't speak of any prior knowledge about the noise and it can be essentailly ignored for the Markov chain analysis. 

#### Is $\{\tilde{X}_n\}_{n \in \mathcal{N}}$ a Markov chain?

Yes. In this case, in order to satify the Markov property, the distribution of the next state $\tilde{X}_{n+1}$ must depend only on the current state, now defined as $\tilde{X}_{n}=(X_n^T, Z_n^T)$. 

Again, according to the given model, to determine the next state we must know $X_n$, $Z_n$ and mutually independent $W_{n+1}$. Thus knowing $X_n$ and $Z_n$ is technically enough to determine the next state's probability distribution. 

Since now knowledge of both $X_n$ and $Z_n$ is coupled within $\tilde{X}_{n}$, knowing $\tilde{X}_{n}$ is indeed enough to determine the probability distribution of $\tilde{X}_{n+1}$.  

#### Trajectory simulation.
![Figure 1: Trajectory simulation.](plot_45.svg)

The presented figure depicts a reasonable trajectory, that is:
- the path is continuous and smooth,
- the direction changes seem consistent with the simulated driving commands,
- the transition matrix $P$ makes the driving command likely to persist for several steps (0.8 probability), which explains the relatively straight segments of the trajectory between major turns,
- the path is not perfectly straight due to the introduced noise, as expected. 

### Problem 2: Observation model

### Problem 3: Mobility tracking using SMC methods (SIS)
### Problem 4: Mobility tracking using SMC methods (SISR)

### Problem 5: SMC-based model calibration

