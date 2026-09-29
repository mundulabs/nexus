//! Explicit device regression; ordinary unit tests do not request a GPU.
#![cfg(all(feature = "webgpu", feature = "mpm", feature = "rbd"))]

use khal::backend::{Backend, GpuBackend, WebGpu};
use nexus3d::{
    mpm::{
        pipeline::MpmPipeline,
        solver::{BoundaryCondition, Particle, ParticleModel, SimulationParams},
    },
    prelude::{NexusCapacities, NexusState, RbdCoupling},
    rapier::prelude::{ColliderBuilder, RigidBodyBuilder},
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
                    // Divergence may disable a particle while retaining a
                    // finite old position. Do not mistake that for stability.
                    let kinematics = backend
                        .slow_read_vec(mpm.particles.kinematics.buffer())
                        .await
                        .unwrap();
                    assert_eq!(kinematics.len(), 64);
                    assert!(kinematics.iter().all(|k| {
                        k.enabled != 0 && k.velocity.is_finite() && k.mass.is_finite()
                    }));
                    let mass: f64 = kinematics.iter().map(|k| f64::from(k.mass)).sum();
                    assert!((mass - 64.0).abs() < 0.001, "Particle mass changed: {mass}");
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

#[test]
#[ignore = "requires a native GPU; run with --ignored --nocapture"]
fn particle_impulse_is_mass_scaled_and_consumed_once() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .unwrap();
        let backend = GpuBackend::WebGpu(WebGpu::from_device(instance, adapter, device, queue));
        let pipeline = MpmPipeline::new(&backend).unwrap();
        for cpic in [false, true] {
            for impulse in [-0.5, 0.0, 0.5] {
                let mut state = NexusState::new(
                    NexusCapacities::default()
                        .mpm_particles(64)
                        .mpm_grid_size(4096),
                );
                state
                    .set_mpm_params(
                        &backend,
                        SimulationParams {
                            gravity: Vector::ZERO,
                            dt: 1.0 / 60.0,
                        },
                        0.2,
                    )
                    .unwrap();
                // A real distant body exercises CPIC without touching the patch.
                state.insert_rigid_body(
                    RigidBodyBuilder::fixed()
                        .translation(Vector::new(0.0, -10.0, 0.0))
                        .build(),
                    ColliderBuilder::cuboid(20.0, 0.2, 20.0).build(),
                    RbdCoupling::MpmOneWay(BoundaryCondition::separate(0.0)),
                );
                let mut particles = Vec::new();
                for x in 0..4 {
                    for y in 0..4 {
                        for z in 0..4 {
                            let mut p = Particle::new(
                                Vector::new(
                                    x as f32 * 0.1 - 0.15,
                                    0.8 + y as f32 * 0.1,
                                    z as f32 * 0.1 - 0.15,
                                ),
                                0.05,
                                2000.0,
                                ParticleModel::fluid(20000.0, 7.0, 0.01),
                            );
                            p.dynamics.force_dt = Vector::new(impulse, 0.0, 0.0);
                            particles.push(p);
                        }
                    }
                }
                state.add_particles(&backend, particles).unwrap();
                state.finalize(&backend).await.unwrap();
                state.set_mpm_use_cpic(cpic);
                let mpm = state.mpm.as_mut().unwrap();
                assert!(!mpm.bodies.is_empty());
                mpm.write_substep_params(&backend, 8).unwrap();
                for frame in 1..=60 {
                    let mut encoder = backend.begin_encoding();
                    for _ in 0..8 {
                        pipeline
                            .encode_step(&backend, &mut encoder, mpm, None)
                            .unwrap();
                    }
                    backend.submit(encoder).unwrap();
                    if frame == 1 || frame == 60 {
                        let kinematics = backend
                            .slow_read_vec(mpm.particles.kinematics.buffer())
                            .await
                            .unwrap();
                        let positions = backend
                            .slow_read_vec(mpm.particles.positions.buffer())
                            .await
                            .unwrap();
                        let velocity = kinematics.iter().map(|k| k.velocity.x).sum::<f32>() / 64.0;
                        let x = positions.iter().map(|p| p.pt.x).sum::<f32>() / 64.0;
                        let expected_velocity = impulse / 2.0;
                        assert!(
                            (velocity - expected_velocity).abs() < 0.001,
                            "cpic={cpic} impulse={impulse} frame={frame}: velocity {velocity}"
                        );
                        assert!(
                            (x - expected_velocity * frame as f32 / 60.0).abs() < 0.001,
                            "cpic={cpic} impulse={impulse} frame={frame}: x {x}"
                        );
                        assert!(kinematics.iter().all(|k| k.enabled != 0
                            && k.force_dt == Vector::ZERO
                            && (k.mass - 2.0).abs() < 0.001));
                    }
                }
            }
        }
    });
}
