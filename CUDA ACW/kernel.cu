#include "kernel.cuh"

#include <cmath>

__global__ void computeSpringForces(PointMass* points, Spring* springs, float2* forces) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    Spring s = springs[i];
    PointMass p1 = points[s.point1];
    PointMass p2 = points[s.point2];

    float2 delta = make_float2(p2.position.x - p1.position.x, p2.position.y - p1.position.y);
    float distance = sqrt(delta.x * delta.x + delta.y * delta.y);
    float magnitude = s.stiffness * (distance - s.restLength);
    float2 force = make_float2(magnitude * delta.x / distance, magnitude * delta.y / distance);

    atomicAdd(&forces[s.point1].x, force.x);
    atomicAdd(&forces[s.point1].y, force.y);
    atomicAdd(&forces[s.point2].x, -force.x);
    atomicAdd(&forces[s.point2].y, -force.y);
}

__global__ void computeDampingForces(PointMass* points, float dampingCoeff, float2* forces) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float2 velocity = points[i].velocity;
    float2 dampingForce = make_float2(-dampingCoeff * velocity.x, -dampingCoeff * velocity.y);
    atomicAdd(&forces[i].x, dampingForce.x);
    atomicAdd(&forces[i].y, dampingForce.y);
}

__global__ void applyGravityAndExternalForces(PointMass* points, float2* forces, float gravity) /* float externalMagnitude */ {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float2 gravityForce = make_float2(0.0f, -gravity * points[i].mass);
    //float2 externalForce = make_float2((rand() % 200 - 100) / 100.0f * externalMagnitude, (rand() % 200 - 100) / 100.0f * externalMagnitude);
    atomicAdd(&forces[i].x, gravityForce.x /*+ externalForce.x*/);
    atomicAdd(&forces[i].y, gravityForce.y /*+ externalForce.y*/);
}

__global__ void computeAccelerations(PointMass* points, float2* forces, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    points[i].velocity.x += (forces[i].x / points[i].mass) * dt;
    points[i].velocity.y += (forces[i].y / points[i].mass) * dt;
}

__global__ void updatePositions(PointMass* points, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    points[i].position.x += points[i].velocity.x * dt;
    points[i].position.y += points[i].velocity.y * dt;
}

__global__ void updateVelocities(PointMass* points, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    points[i].velocity.x = (points[i].position.x - points[i].position.x) / dt;
    points[i].velocity.y = (points[i].position.y - points[i].position.y) / dt;
}

