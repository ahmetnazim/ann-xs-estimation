"""Model ladder: M-null, A_J, M0-M6 and the M5 ablations.

ARCHITECTURE RULE: no batch-coupled layer (e.g. BatchNorm in training mode) upstream of Phi;
LayerNorm only (proofs Proposition 3.4 scope; P3-17). Models read the frozen fold table and
never create folds (protocol 3.3); every learned transform is fitted on training folds (3.4).
Owning phase: 6. Placeholder (P4-D8 (a)); no numerical, physics or ML logic yet.
"""
