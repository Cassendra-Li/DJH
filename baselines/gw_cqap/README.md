# GW_CQAP Baseline Reference

**Source**: Gromov-Wasserstein and optimal transport: from assignment problems to probabilistic numeric
**Authors**: Iman Seyedi, Antonio Candelieri, Enza Messina, Francesco Archetti
**Code**: https://github.com/iman-ie/GW_CQAP

## Key Points

- Establishes the connection between Quadratic Assignment Problems and GW optimal transport
- Proposes **GW_MultiInit** (multiple random initializations) → reduces GW cost gap from 18.12% to 1.78%
- EGW parameter tuning: ε = 0.8 optimal for CQAP (accuracy-speed balance)
- FGW α analysis: structure info (α=0.7) more important than feature info

## How We Use This

1. **GW_MultiInit** strategy → directly used in our GWAlignment module
2. **EGW ε=0.8** → initial value for entropy-regularized GW variants
3. **FGW α analysis** → reference for our α sensitivity experiments
4. **Scalability analysis framework** → adopt their accuracy-runtime trade-off visualization
