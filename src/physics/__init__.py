"""Physics layer Phi: hard Manning + continuity map from (W, r, n, Q_E, S) to geometry.

Sources: proofs sections 2-3. Rules: Phi acts row-wise (Proposition 3.4); SI units only
(Propositions 1.2, 3.5); CPU float64 (protocol 4.2, O-8).
Owning phase: 6. Placeholder (P4-D8 (a)); no numerical, physics or ML logic yet.
"""
