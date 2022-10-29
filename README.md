### Investigation into the Longstaff-Schwartz Method used for American Option Pricing

Within this repo is the LSM algorithm utilizing Brownian Bridges, inspired by the original paper Evaluating the Longstaff-Schwartz Method for Pricing American Options by William Gustaffson. 

High-level the algorithm
- Iteratively samples observations backward in time using the theoretical results of Brownian bridges.
- Estimates continuatiion values via least squares on underlying realizations and discounted payoffs
- Generate the exercise boundary from sample mean conditional on the derivative exercising early. 
- Compares to naive approach of generating exercise boundary by local extrema 
