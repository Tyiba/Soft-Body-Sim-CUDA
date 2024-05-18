#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

struct PointMass {
    float2 position;
    float2 velocity;
    float mass;
};

struct Spring {
    int point1;
    int point2;
    float restLength;
    float stiffness;
};

__global__ void computeSpringForces(PointMass* points, Spring* springs, float2* forces, int numSprings);
__global__ void computeDampingForces(PointMass* points, float dampingCoeff, float2* forces, int numPoints);
__global__ void applyGravityAndExternalForces(PointMass* points, float2* forces, float gravity, float externalMagnitude, int numPoints);
__global__ void computeAccelerations(PointMass* points, float2* forces, int numPoints);
__global__ void updatePositions(PointMass* points, float dt, int numPoints);
__global__ void updateVelocities(PointMass* points, float dt, int numPoints);

#endif // KERNEL CUDA HEADER
