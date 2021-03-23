<img src="./tinyDA.png" width="500">

# tinyDA
Delayed Acceptance MCMC sampler with finite-length subchain sampling. 
This is intended as a simple, lightweight implementation, with minimal dependencies, e.g. nothing beyond the SciPy stack.

## This is still a work in progress.

## Proposals
* Random Walk Metropolis Hastings (RWMH) - Metropolis et al. (1953), Hastings (1970).
* preconditioned Crank-Nicolson (pCN) - Cotter et al. (2013).
* Adaptive Metropolis (AM) - Haario et al. (2001).
* Adaptive pCN - Hu et al. (2016).

## Adaptive Error Models
* State independent - Cui et al. (2012)
* State dependent - Cui et al. (2018)

## Dependencies:
* NumPy
* SciPy
* tqdm
