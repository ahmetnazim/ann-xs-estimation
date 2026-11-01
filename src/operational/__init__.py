"""OPS-1 operational evaluation: pre-registered stage-response test of predicted cross-sections.

Post-release extension of v1.0.0 (docs/4_operational_stage_response.md). The package reads the
frozen v1.0.0 predictions and HYDROSWOT, and never modifies a v1.0.0 artefact.

Modules
-------
config
    Schema and loader for ``params_operational.yaml``.
rating
    Calibrated-roughness Dingman-Manning synthetic rating curve (OPS-E1, Proposition OPS-1).
observations
    Datum-free observed stage and width responses at the S-ID gauges.
evaluate
    Per-model responses, errors, bootstrap contrasts and the result bundle.
"""
