//! Explicit device regression; ordinary unit tests do not request a GPU.
#![cfg(all(feature = "webgpu", feature = "mpm", feature = "rbd"))]

use khal::backend::{Backend, GpuBackend, WebGpu};
use nexus3d::{
    mpm::{
        pipeline::MpmPipeline,
        solver::{Particle, ParticleModel, SimulationParams},
    },
    prelude::{NexusCapacities, NexusState},
    rbd::math::Vector,
};

#[test]
#[ignore = "requires a native GPU; run with --ignored --nocapture"]
fn empty_colliders_preserve_free_fall_in_both_requested_modes() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        eprintln!("MPM empty-world regression: {:?}", adapter.get_info());
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .unwrap();
        let backend = GpuBackend::WebGpu(WebGpu::from_device(instance, adapter, device, queue));
        let pipeline = MpmPipeline::new(&backend).unwrap();
        for requested_cpic in [true, false] {
            for gravity in [0.0, -9.81] {
                let mut baseline: Option<Vec<Vector>> = None;
                for batched in [false, true] {
                    let mut state = NexusState::new(
                        NexusCapacities::default()
                            .mpm_particles(64)
                            .mpm_grid_size(4096),
                    );
                    state
                        .set_mpm_params(
                            &backend,
                            SimulationParams {
                                gravity: Vector::new(0.0, gravity, 0.0),
                                dt: 1.0 / 60.0,
                            },
                            0.2,
                        )
                        .unwrap();
                    let mut particles = Vec::new();
                    for x in 0..4 {
                        for y in 0..4 {
                            for z in 0..4 {
                                particles.push(Particle::new(
                                    Vector::new(
                                        x as f32 * 0.1 - 0.15,
                                        0.8 + y as f32 * 0.1,
                                        z as f32 * 0.1 - 0.15,
                                    ),
                                    0.05,
                                    1000.0,
                                    ParticleModel::fluid(20000.0, 7.0, 0.01),
                                ));
                            }
                        }
                    }
                    state.add_particles(&backend, particles).unwrap();
                    state.finalize(&backend).await.unwrap();
                    state.set_mpm_use_cpic(requested_cpic);
                    let mpm = state.mpm.as_mut().unwrap();
                    assert!(mpm.bodies.is_empty());
                    mpm.write_substep_params(&backend, 8).unwrap();
                    for _ in 0..60 {
                        if batched {
                            let mut encoder = backend.begin_encoding();
                            for _ in 0..8 {
                                pipeline
                                    .encode_step(&backend, &mut encoder, mpm, None)
                                    .unwrap();
                            }
                            backend.submit(encoder).unwrap();
                        } else {
                            for _ in 0..8 {
                                pipeline.step(&backend, mpm, None).unwrap();
                            }
                        }
                    }
                    let positions = backend
                        .slow_read_vec(mpm.particles.positions.buffer())
                        .await
                        .unwrap();
                    assert_eq!(positions.len(), 64);
                    assert!(positions.iter().all(|p| p.pt.is_finite()));
                    let mean_y = positions.iter().map(|p| p.pt.y).sum::<f32>() / 64.0;
                    let expected = 0.95 + gravity * (1.0_f32 / 480.0).powi(2) * 480.0 * 481.0 * 0.5;
                    assert!(
                        (mean_y - expected).abs() < 0.005,
                        "cpic={requested_cpic}, gravity={gravity}: mean {mean_y}, expected {expected}"
                    );
                    let points: Vec<Vector> = positions.iter().map(|p| p.pt).collect();
                    if let Some(expected) = baseline.as_ref() {
                        for (a, b) in points.iter().zip(expected) {
                            assert!(
                                (*a - *b).length() < 0.0001,
                                "Batching changed final particle position"
                            );
                        }
                    } else {
                        baseline = Some(points);
                    }
                }
            }
        }
    });
}
