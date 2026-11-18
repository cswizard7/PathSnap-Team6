# Evaluator-testing partition (final evaluation only)

This folder is intentionally empty during development. On evaluation day, copy
the 10 evaluator trip folders here, exactly as distributed, e.g.

    data/evaluator_test/00000015/00000015.track
    data/evaluator_test/00000015/00000015.nodes
    data/evaluator_test/00000015/00000015.arcs
    data/evaluator_test/00000015/00000015.route   (read server-side only, for scoring)

No code in this repository trains on, tunes on or inspects this folder; only
the API reads it, at request time.
