#include <chrono>
#include <glad.h>
#include <glfw3.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <windows.h>

#include "rendering.h"

#pragma region Settings

constexpr int WIDTH = 500; //oarticles x 
constexpr int HEIGHT = 500; // particles y
constexpr int NUM_PARTICLES = WIDTH * HEIGHT;
constexpr float GRAVITY = -9.81f; // Standard gravity value
constexpr float RENDERSCALE = 0.5f;
constexpr float SPRING_REST_LENGTH = 1.0f / HEIGHT * 2.0f; // Distance between particles in the window
constexpr float SPRING_COEFFICIENT = 25.0f; // Adjusted for window size
constexpr float DAMPING_COEFFICIENT = 10.0f; // Adjusted for window size
constexpr float EXTERNAL_MAGNITUDE = 0.01f; // Adjusted for window size
constexpr float MASS = 0.01f;
constexpr float DELTA_TIME = 0.01f;
bool gravityEnabled = true;

struct Particle {
    float2 position;
    float2 velocity;
    float2 force;
    bool fixed;
};

struct Grid {
    int width;
    int height;
    Particle* particles;
    int* neighbors;
};

#pragma endregion

#pragma region Operators for Float2

__device__ float length(const float2& a) {
    return sqrtf(a.x * a.x + a.y * a.y);
}
__device__ float2 operator+(const float2& a, const float2& b) {
    return make_float2(a.x + b.x, a.y + b.y);
}
__device__ float2 operator-(const float2& a, const float2& b) {
    return make_float2(a.x - b.x, a.y - b.y);
}
__device__ float2 operator*(const float2& a, const float2& b) {
    return make_float2(a.x * b.x, a.y * b.y);
}
__device__ float2 operator*(const float2& a, float b) {
    return make_float2(a.x * b, a.y * b);
}
__device__ float2 operator/(const float2& a, float b) {
    return make_float2(a.x / b, a.y / b);
}
__device__ float2 operator*(float b, const float2& a) {
    return make_float2(a.x * b, a.y * b);
}
__device__ float2 operator/(float b,const float2& a) {
    return make_float2(a.x / b, a.y / b);
}
__device__ void operator+=(float2& a, const float2& b) {
    a.x += b.x;
    a.y += b.y;
}
__device__ void operator-=(float2& a, const float2& b) {
    a.x -= b.x;
    a.y -= b.y;
}
#pragma endregion


__global__ void initializeGrid(Particle* particles, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        float xPos = (x - width / 2.0f) / (width / 2.0f);
        float yPos = (y - height / 2.0f) / (height / 2.0f);
        particles[idx].position = make_float2(xPos, yPos);
        particles[idx].velocity = make_float2(0.0f, 0.0f);
        particles[idx].force = make_float2(0.0f, 0.0f);
        particles[idx].fixed = (x == 0 && y == height - 1) || (x == width - 1 && y == height - 1); // Fix top row and corners
    }
}

__global__ void calculateNeighbors(int* neighbors, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        int n = 0;

        if (x > 0) neighbors[idx * 4 + n++] = idx - 1; // Left neighbor
        if (x < width - 1) neighbors[idx * 4 + n++] = idx + 1; // Right neighbor
        if (y > 0) neighbors[idx * 4 + n++] = idx - width; // Up neighbor
        if (y < height - 1) neighbors[idx * 4 + n++] = idx + width; // Down neighbor

        for (int i = n; i < 4; ++i) {
            neighbors[idx * 4 + i] = -1; // Mark unused neighbor slots
        }
    }
}


__global__ void applyForces(Particle* particles, int* neighbors, int width, int height, float deltaTime, bool gravityEnabled) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        Particle& p = particles[idx];

        if (!p.fixed) {

            float2 totalForce = make_float2(0.0f, 0.0f);
            if (gravityEnabled) 
                totalForce.y += GRAVITY * MASS;

            float2 currentVelocity = p.velocity;
            float2 oldposition = p.position;

            for (int i = 0; i < 4; ++i) {
                int neighborIdx = neighbors[idx * 4 + i];

                if (neighborIdx != -1) {
                    Particle& neighbor = particles[neighborIdx];

                    float2 delta = neighbor.position - p.position;
                    float dist = length(delta);
                    float magnitude = SPRING_COEFFICIENT * (dist - SPRING_REST_LENGTH);
                    float2 force = magnitude * delta / dist;
                    totalForce += force;

                    atomicAdd(&neighbor.force.x, force.x);
                    atomicAdd(&neighbor.force.y, force.y);
                }
            }

            // Apply damping
            float2 dampingForce = -DAMPING_COEFFICIENT * currentVelocity;
            totalForce += dampingForce;
            // Apply random external force
            curandState state;
            curand_init(0, idx, 0, &state);
            float2 randomForce = make_float2(
                curand_uniform(&state) * 2.0f - 1.0f,
                curand_uniform(&state) * 2.0f - 1.0f
            ) * EXTERNAL_MAGNITUDE;
            //printf("Random State %f, %f, \n", randomForce.x, randomForce.y);
            totalForce += randomForce;
            // Update particle force
            p.force = totalForce;
            float2 a = totalForce / MASS;
            //Update Point pos and vel
            p.position = p.position + currentVelocity * deltaTime + 0.5f * a * deltaTime * deltaTime;
            p.velocity += (p.position - oldposition) * deltaTime;
            // Reset force
            p.force = make_float2(0.0f, 0.0f);

        }
    }
}



/*
__global__ void updateParticles(Particle* particles, int numParticles, float deltaTime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        Particle& p = particles[idx];
        if (!p.fixed) {
                        // Update velocity
            p.velocity += ( p.position - oldposition ) * deltaTime;

            // Reset force
            p.force = make_float2(0.0f, 0.0f);

            // Debugging: Print updated positions and velocities
           //printf("Particle %d: Position (%f, %f) Velocity (%f, %f)\n", idx, p.position.x, p.position.y, p.velocity.x, p.velocity.y);
        }
    }
}
*/



Particle* d_particles;
int* d_neighbors;
Particle h_particles[NUM_PARTICLES];

void initializeSimulation() {
    cudaMalloc(&d_particles, NUM_PARTICLES * sizeof(Particle));
    cudaMalloc(&d_neighbors, NUM_PARTICLES * 4 * sizeof(int));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);
    initializeGrid << <numBlocks, threadsPerBlock >> > (d_particles, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    calculateNeighbors << <numBlocks, threadsPerBlock >> > (d_neighbors, WIDTH, HEIGHT);
    cudaDeviceSynchronize();
}

cudaEvent_t start, stop;
float totalTime = 0.0f;
int iterations = 0;
float milliseconds = 0;

void updateSimulation() {
    int numThreads = 256;
    int numBlocksParticles = (NUM_PARTICLES + numThreads - 1) / numThreads;


    cudaEventRecord(start);
    applyForces << <numBlocksParticles, numThreads >> > (d_particles, d_neighbors, WIDTH, HEIGHT, DELTA_TIME, gravityEnabled);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    
    cudaEventElapsedTime(&milliseconds, start, stop);
    totalTime += milliseconds;
    iterations++;

    //updateParticles << <numBlocksParticles, numThreads >> > (d_particles, NUM_PARTICLES, DELTA_TIME);
    //cudaDeviceSynchronize();
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS && (key == GLFW_KEY_G)) {
        gravityEnabled = !gravityEnabled;
        printf("Gravity toggled: %s\n", gravityEnabled ? "On" : "Off");
    }
}

int main() {
    initializeSimulation();

    initializeOpenGL();

    GLFWwindow* window = glfwGetCurrentContext();
    glfwSetKeyCallback(window, keyCallback);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto startTime = std::chrono::high_resolution_clock::now();
    while (!glfwWindowShouldClose(window)) {
        auto currentTIme = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsedTime = currentTIme - startTime;
        if (elapsedTime.count() >= 10.0f) break;

        updateSimulation();

        cudaMemcpy(h_particles, d_particles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);

        std::vector<float2> positions(NUM_PARTICLES);
        for (int i = 0; i < NUM_PARTICLES; ++i) {
            positions[i] = h_particles[i].position;
        }

        renderGrid(positions.data(), WIDTH, HEIGHT, RENDERSCALE);
        glfwPollEvents();
    }

    float average = totalTime / iterations;
    printf("Average Time for %d x %d grid: %f µs \n", WIDTH, HEIGHT, average * 1000.0f);

    //average times

    cudaFree(d_particles);
    cudaFree(d_neighbors);
    glfwTerminate();
    return 0;
}