# textkit

A tiny utility library used as the evaluation target for mender. The eval
generator injects single-token bugs into `src/textkit/` and asks the agent to
fix them. `tests/` is visible to the agent; `../hidden_tests/` is not.
