"""Thin CLI: score stored predictions and run the pre-registered procedures.

Sources: protocol sections 4.3-4.7.
Owning phase: 8. The library (:mod:`src.evaluation`) is implemented and verified on synthetic
inputs in Phase 7 (P7-D1 (a)); main() stays locked until the Phase 7 gate so that no outer-test
prediction is scored and no H1-H4 verdict is produced in Phase 7.
"""

LOCK_MESSAGE = "Phase 8: scoring locked until the Phase 7 gate"


def main() -> None:
    """Run this entry point; locked in Phase 7 (P7-D1 (a))."""
    raise NotImplementedError(LOCK_MESSAGE)


if __name__ == "__main__":
    main()
