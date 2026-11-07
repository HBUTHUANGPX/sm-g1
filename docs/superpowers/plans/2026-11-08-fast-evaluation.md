# Fixed fast G1 evaluation implementation plan

Goal: run both existing all-40 protocols from one reusable command and automatically produce validated human-readable results, with measured Newton speedup.

Architecture: preserve original ManagerBased physics/training and protocol v2. Evaluation-only grouped PD uses the same per-joint parameters and RNG draw order with vectorized independent delay storage. Keep a reference execution mode for differential verification. Add progress and duration reports, plus a fixed runner using an ignored local configuration for machine paths.

Alternatives considered: only reduce host synchronization (profiling shows minor share); concurrent profile processes (additional contention, does not remove the dominant 29-actuator loop); grouped evaluation actuators (chosen based on profiling). Do not shorten clips or relax validation.

- [ ] Implement and test a joint-wise delay adapter against 29 original buffers including asynchronous resets, saturation and seeded draw order. Add grouped evaluation actuator configuration and reference fallback; benchmark both on Newton before making grouped default.
- [ ] Add periodic evaluation progress and timing, keeping original trajectory boundaries and metrics; automate dual-profile report generation with exact input/result validation.
- [ ] Add portable local-config entry point with checkpoint selection, immutable result dirs, source/checkpoint fingerprints and reuse validation. Keep private machine paths out of Git.
- [ ] Run CPU regression checks and actual Newton A/B acceptance on checkpoint6600; commit/push each validated unit. Run complete optimized all40 dual evaluation and publish benchmark/protocol limitations through PR.
