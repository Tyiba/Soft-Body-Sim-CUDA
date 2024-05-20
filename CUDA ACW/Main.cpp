
#include <cuda_runtime_api.h>
#include <iostream>
#include <ostream>
#include <vector>

#include "glad.h"
#include <cuda_gl_interop.h>
#include "glfw3.h"
#include "kernel.cuh"




void processInput(GLFWwindow* window);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void Init();
void Update();
void Render();
void Reset();

// settings
constexpr unsigned int SCR_WIDTH = 800;
constexpr unsigned int SCR_HEIGHT = 600;
GLuint vbo;
cudaGraphicsResource* cudaVboResource;
PointMass* d_points;
Spring* d_springs;

const int N = 10;  // Number of points along one dimension
const int M = 10;  // Number of points along the other dimension
constexpr float springRelaxDistance = 1.0f;
constexpr float springCoeff = 10.0f;
bool gravity = false;


//Visualization

//  0 --- 0 --- 0 --- 0 <- Point Mass
//  |     |     |     |
//  |     |     |     |
//  0 --- 0 --- 0 --- 0
//  |     |     |     | <- Springs
//  |     |     |     |
//  0 --- 0 --- 0 --- 0
//  |     |     |     |
//  |     |     |     |
//  0 --- 0 --- 0 --- 0

int main() {
#pragma region Window INIT
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "CUDA SOFT BODY SIMULATION", nullptr, nullptr);
	if (window == nullptr) {
        std::cout << "Failed to create GLFW window" << '\n';
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
    {
        std::cout << "Failed to initialize GLAD" << '\n';
        return -1;
    }
#pragma endregion

    Init();

    while (!glfwWindowShouldClose(window))
    {
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        //Input -> Update -> Render -> Events + Buffer Swap

        processInput(window);

        Update();

        Render();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

void Init()
{
    // OpenGL VBO creation
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, N * M * sizeof(PointMass), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Register VBO with CUDA
    cudaGraphicsGLRegisterBuffer(&cudaVboResource, vbo, cudaGraphicsMapFlagsWriteDiscard);

    // Allocate device memory for point masses and springs
    cudaMalloc(&d_points, N * M * sizeof(PointMass));
    cudaMalloc(&d_springs, ((N - 1) * M + N * (M - 1)) * sizeof(Spring));

    // Initialize point masses and springs on the host
    std::vector<PointMass> h_points(N * M);
    std::vector<Spring> h_springs((N - 1) * M + N * (M - 1));

    // Initialize point masses
    for (int y = 0; y < M; ++y) {
        for (int x = 0; x < N; ++x) {
            int index = y * N + x;
            h_points[index].position = make_float2(x * springRelaxDistance, y * springRelaxDistance);
            h_points[index].velocity = make_float2(0.0f, 0.0f);
            h_points[index].mass = 0.01f;
        }
    }

    // Initialize springs
    int springIndex = 0;
    for (int y = 0; y < M; ++y) {
        for (int x = 0; x < N - 1; ++x) {
            int index = y * N + x;
            h_springs[springIndex++] = { index, index + 1, springRelaxDistance, springCoeff };
        }
    }
    for (int y = 0; y < M - 1; ++y) {
        for (int x = 0; x < N; ++x) {
            int index = y * N + x;
            h_springs[springIndex++] = { index, index + N, springRelaxDistance, springCoeff };
        }
    }

    // Copy data from host to device
    cudaMemcpy(d_points, h_points.data(), N * M * sizeof(PointMass), cudaMemcpyHostToDevice);
    cudaMemcpy(d_springs, h_springs.data(), ((N - 1) * M + N * (M - 1)) * sizeof(Spring), cudaMemcpyHostToDevice);
}

void Update()
{
	
}

void Render()
{
	
}

void Reset()
{
	
}

void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
        Reset();
    if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS)
        gravity = true;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}
