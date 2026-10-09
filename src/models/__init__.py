"""Model ladder: M-null, A_J, M0-M6 and the M5 ablations.

ARCHITECTURE RULE: no batch-coupled layer (e.g. BatchNorm in training mode) upstream of Phi;
LayerNorm only (proofs Proposition 3.4 scope; P3-17). Models read the frozen fold table and
never create folds (protocol 3.3); every learned transform is fitted on training folds (3.4).
Owning phase: 6. This package imports nothing (P6-D7 (a)): ``baselines`` stays torch-free and
the torch modules (``backbone``, ``heads``, ``losses``, ``m2_mlp``-``m6_two_stage``, ``ensemble``)
never load xgboost (ADL-051; import tests in ``tests/unit/test_models_imports.py``).
"""
