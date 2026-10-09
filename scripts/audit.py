"""Thin CLI: run the spatial and network dependence audit.

Sources: protocol section 3.6.
Owning phase: 8. The library (:mod:`src.evaluation.dependence_audit`) is implemented and
verified on synthetic inputs in Phase 7 (P7-D1 (a)); main() stays locked until the Phase 7 gate
so that no audit of real residuals runs before the harness is signed off.
"""

LOCK_MESSAGE = "Phase 8: scoring locked until the Phase 7 gate"


def main() -> None:
    """Run this entry point; locked in Phase 7 (P7-D1 (a))."""
    raise NotImplementedError(LOCK_MESSAGE)


if __name__ == "__main__":
    main()
