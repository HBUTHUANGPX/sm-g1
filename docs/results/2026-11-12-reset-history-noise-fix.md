# Reset history noise consistency fix — 2026-11-12

Scope: fix only the mismatch identified in the official-code audit. Real collection already repeats one noisy reset observation. Imagined initialization now reuses the noisy observation of each repeated replay frame, including the current seed at episode step zero. It derives source slots from existing episode-step metadata, so no replay/checkpoint schema or training configuration changes are required.

The original noise draw shape and RNG consumption are preserved. Nonpadding slots retain their exact old draws; repeated prefix slots share the rightmost copy of the oldest available frame. This also covers a replay ring edge inside an ongoing episode. Noise is still sampled for imagined histories rather than replaying the historical random realization; consistency means matching the per-frame noise/padding semantics, not reusing the exact historical random values.

Validation uses the production CollectionMixin.collect history update (stopping before policy/simulation work), production imagined-history initializer, MDP.add_noise, and ReplayBuffer.sample_starts. Tests cover history lengths 1/2/10, asynchronous episode ages 0/1/4/12, reset and partial padding, fully populated histories, wrapped replay rings, retained histories beginning mid-episode, unchanged current-frame noise, and unchanged RNG state after sampling. Under identical per-frame noise draws the real and imagined histories match exactly.

Before the semantic fix: four regression cases failed and two unchanged-history cases passed. After the fix and addition of two real-replay cases: eight regression cases pass. Full G1 CPU suite: **178 passed, 4 runtime-dependent skips**. The architecture fingerprint correctly detected the changed initializer call; only that authorized method fingerprint was updated with its prior digest/reason retained, and the new helper was added to the contract. Other algorithm/config fingerprints remain unchanged.

No GPU simulation or training was launched. This is a deterministic history-construction regression fix; training-quality impact is not measured. No other official/local algorithm differences were changed.
