# Twelve-worker pilot queues

Subsequent development, reserved and representative forecast pilots use the existing shared native scheduler through `pilot_queue.py`, with12worker slots and a source-bound selected-task spec. This replaces the serial pilot execution path; completed serial source/results remain preserved evidence.

A24-forecast orchestration check observed12simultaneously active forecast workers and12distinct native worker PIDs. All24whole native CRPS loss vectors matched the completed incumbent bit-for-bit, with0failures. This is scheduler verification, not a new model, full-panel admission, independent validation or runtime speedup claim. The original full320245-attempt campaign also used12workers.
