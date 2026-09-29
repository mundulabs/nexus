# Mundus GPU integration branch

This company fork tracks upstream Nexus. `main` retains upstream history;
`gpu-stability` starts at release 0.5.0 (`5b05f4a`) and carries
the empty-collider fix. Keep downstream engine changes small and preserve
upstream license and attribution notices. Mundus-specific nodes, registry
descriptors, rendering and preparation scheduling belong in Mundus.

The fix selects plain MPM when no coupled bodies exist, skips empty collision
bindings and removes unused body arguments from the plain G2P kernel. It
does not insert a collider or change the material/contact law. Both host and
shader declarations must change together. The umbrella simulation API also
propagates MPM upload/step errors instead of reporting success after discarding
them. An error does not roll back previously submitted GPU work; reject the
candidate and prepare a replacement. The shared source also builds 2D,
but only 3D Metal execution has been observed so far.

`MpmPipeline::encode_step` also lets a host record several fixed substeps into
one command encoder, then submit once. `step` retains its original convenience
behaviour. This avoids one queue submission per substep without changing dt or
the number of simulated steps. The opt-in regression compares every final
particle position between the two submission paths for gravity on/off and
both requested boundary modes. This is not CUDA graph capture or a claim of
allocation-free command recording.

The added GPU regression is opt-in:

```sh
cargo test --release -p nexus3d --features rbd,mpm --test mpm_empty_world -- --ignored --nocapture
```

The local qualification uses Rust 1.98.1, cargo-gpu 0.10.0-alpha.1 and the
Rust-GPU nightly-2026-04-11 toolchain. Install the shader toolchain serially before
building. The downstream Mundus probe pins wgpu 30.0.1 and checks gravity,
contacts, fresh-state preservation and startup timing on a shared device with
rendering. Its tested source and logs are recorded in Mundus's physics evidence.
Mundus should depend on an exact reviewed commit, not this moving branch.

On 29 September 2026, the explicit test passed on Apple M2 Max / Metal across
eight combinations (gravity on/off, requested CPIC on/off, per-step/batched
submission). Every particle stayed finite, free fall matched the discrete
integration result, and batching preserved final positions within 0.0001.
Formatting and Clippy with warnings denied passed for the three changed 3D
crates and their targets. This is bounded 64-particle evidence, not a large-scene
or full-platform qualification.

Known qualification gaps include startup latency outliers, live empty/nonempty
transitions, resource overflow, long-running coupled worlds, NVIDIA hardware,
native host integration and surfaced ferrofluid/reference fidelity. A passing
empty-world test does not qualify those capabilities.

The force-input follow-up fixes a separate defect: `Kinematics::force_dt` was
stored and reset but never consumed by P2G. Plain MPM now adds that impulse to
momentum; CPIC converts it to velocity by particle mass before its compatibility
handling. Particle update clears it as before. Hosts must republish a sustained
force each substep; one impulse must not be repeated automatically. Zero-force
worlds retain their existing path. The opt-in regression checks positive, zero
and negative impulses with non-unit mass in both actual boundary modes, after
one frame and one second, including cleared input and preserved particle mass.
It passed on Apple M2 Max/Metal. This qualifies an input, not a built-in magnetic
or surface-tension model. The free-fall checks also reject disabled particles
and lost mass instead of accepting finite but frozen positions.
