## Log_lobotomy
Used CFR+ with possibly buggy cum_strat smoothing (divided by 1e6 anytime sum(cum_strat) > 1e12). This may be what caused catastrophic forgetting?

## Current run
CFR+ to update cum_regret, Vanilla CFR to update cum_strat