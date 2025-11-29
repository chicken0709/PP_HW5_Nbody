#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <hip/hip_runtime.h>

#define HIP_CHECK(cmd) \
{ \
    hipError_t error = cmd; \
    if (error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

namespace param {
    const int n_steps = 200000;
    const double dt = 60;
    const double eps = 1e-3;
    const double G = 6.674e-11;
    const double planet_radius = 1e7;
    const double missile_speed = 1e6;
    inline double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
}

// Device constants
__constant__ double d_dt;
__constant__ double d_eps;
__constant__ double d_G;
__constant__ double d_planet_radius;
__constant__ double d_missile_speed;

__global__ void check_min_dist(int planet, int asteroid, double* qx, double* qy, double* qz, double* min_dist) {
    if (threadIdx.x == 0) {
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        double dist = sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < *min_dist) {
            *min_dist = dist;
        }
    }
}

__global__ void compute_forces_update_velocity(int n, double* qx, double* qy, double* qz,
                                        double* vx, double* vy, double* vz,
                                        double* m, double* m0, int* type, double t,
                                        int planet, int asteroid, double* min_dist) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (min_dist != nullptr && i == 0) {
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        double dist = sqrt(dx * dx + dy * dy + dz * dz);
        atomicMin(min_dist, dist);
    }
    
    double ax = 0.0, ay = 0.0, az = 0.0;
    double cur_qx, cur_qy, cur_qz;

    if (i < n) {
        cur_qx = qx[i];
        cur_qy = qy[i];
        cur_qz = qz[i];
    }

    if (i < n && m0 != nullptr && type[i] == 2) {
        double tmp = m0[i];
        m[i] = tmp + 0.5 * tmp * fabs(sin(t / 6000.0));
    }

    __shared__ double s_qx[256];
    __shared__ double s_qy[256];
    __shared__ double s_qz[256];
    __shared__ double s_m[256];

    for (int tile = 0; tile < n; tile += blockDim.x) {
        int idx = tile + threadIdx.x;
        if (idx < n) {
            s_qx[threadIdx.x] = qx[idx];
            s_qy[threadIdx.x] = qy[idx];
            s_qz[threadIdx.x] = qz[idx];
            s_m[threadIdx.x] = m[idx];
        }
        __syncthreads();

        if (i < n) {
            int limit = (tile + blockDim.x > n) ? (n - tile) : blockDim.x;
            for (int j = 0; j < limit; j++) {
                int global_j = tile + j;
                if (i == global_j) continue;

                double mj = s_m[j];
                
                double dx = s_qx[j] - cur_qx;
                double dy = s_qy[j] - cur_qy;
                double dz = s_qz[j] - cur_qz;
                double dist2 = dx * dx + dy * dy + dz * dz + d_eps * d_eps;
                
                double invDist = rsqrt(dist2);
                double invDist3 = invDist * invDist * invDist;
                double f = d_G * mj * invDist3;

                ax += f * dx;
                ay += f * dy;
                az += f * dz;
            }
        }
        __syncthreads();
    }

    if (i < n) {
        vx[i] += ax * d_dt;
        vy[i] += ay * d_dt;
        vz[i] += az * d_dt;
    }
}

__global__ void update_position(int n, double* qx, double* qy, double* qz,
                           double* vx, double* vy, double* vz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        qx[i] += vx[i] * d_dt;
        qy[i] += vy[i] * d_dt;
        qz[i] += vz[i] * d_dt;
    }
}

__global__ void check_hits(int n, int planet, int asteroid, double* qx, double* qy, double* qz, int* type, int step, int* saved_step, int* hit_step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check planet-asteroid collision
    if (i == 0) {
        if (*hit_step == -1) {
            double dx = qx[planet] - qx[asteroid];
            double dy = qy[planet] - qy[asteroid];
            double dz = qz[planet] - qz[asteroid];
            if (dx * dx + dy * dy + dz * dz < d_planet_radius * d_planet_radius) {
                *hit_step = step;
            }
        }
    }

    // Check missile-device hit
    if (i < n && type[i] == 2) {
        if (*saved_step != -1) return;

        double dx = qx[planet] - qx[i];
        double dy = qy[planet] - qy[i];
        double dz = qz[planet] - qz[i];
        double dist = sqrt(dx * dx + dy * dy + dz * dz);
        
        double missile_dist = step * d_dt * d_missile_speed;
        if (missile_dist > dist) {
            atomicCAS(saved_step, -1, step);
        }
    }
}

__global__ void save_state(int n, double* qx, double* qy, double* qz,
                                  double* vx, double* vy, double* vz,
                                  double* m, int* type, int step, int* saved_step,
                                  double* s_qx, double* s_qy, double* s_qz,
                                  double* s_vx, double* s_vy, double* s_vz,
                                  double* s_m, int* s_type) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (*saved_step == step) {
            s_qx[i] = qx[i];
            s_qy[i] = qy[i];
            s_qz[i] = qz[i];
            s_vx[i] = vx[i];
            s_vy[i] = vy[i];
            s_vz[i] = vz[i];
            s_m[i] = m[i];
            s_type[i] = type[i];
        }
    }
}

__global__ void check_hit_and_destroy(int planet, int asteroid, int target_device, double* qx, double* qy, double* qz, 
                                     double* m, int* type, int step, int* hit_step, int* destroyed_step) {
    if (threadIdx.x == 0) {
        if (*hit_step == -1) {
            double dx = qx[planet] - qx[asteroid];
            double dy = qy[planet] - qy[asteroid];
            double dz = qz[planet] - qz[asteroid];
            if (dx * dx + dy * dy + dz * dz < d_planet_radius * d_planet_radius) {
                *hit_step = step;
            }
        }

        if (*destroyed_step == -1) {
            double dx = qx[planet] - qx[target_device];
            double dy = qy[planet] - qy[target_device];
            double dz = qz[planet] - qz[target_device];
            double dist = sqrt(dx * dx + dy * dy + dz * dz);
            
            double missile_dist = step * d_dt * d_missile_speed;
            if (missile_dist > dist) {
                *destroyed_step = step;
                type[target_device] = 3; // destroyed
                m[target_device] = 0.0;
            }
        }
    }
}

struct Data {
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<int> type;
    int n, planet, asteroid;
};

void read_input(const char* filename, Data& ctx) {
    std::ifstream fin(filename);
    fin >> ctx.n >> ctx.planet >> ctx.asteroid;
    ctx.qx.resize(ctx.n); ctx.qy.resize(ctx.n); ctx.qz.resize(ctx.n);
    ctx.vx.resize(ctx.n); ctx.vy.resize(ctx.n); ctx.vz.resize(ctx.n);
    ctx.m.resize(ctx.n); ctx.type.resize(ctx.n);
    
    for (int i = 0; i < ctx.n; i++) {
        std::string type;
        fin >> ctx.qx[i] >> ctx.qy[i] >> ctx.qz[i] 
            >> ctx.vx[i] >> ctx.vy[i] >> ctx.vz[i] 
            >> ctx.m[i] >> type;
        if (type == "planet") ctx.type[i] = 0;
        else if (type == "asteroid") ctx.type[i] = 1;
        else if (type == "device") ctx.type[i] = 2;
        else ctx.type[i] = 3;
    }
}

void write_output(const char* filename, double min_dist, int hit_time_step,
    int gravity_device_id, double missile_cost) {
    std::ofstream fout(filename);
    fout << std::scientific
         << std::setprecision(std::numeric_limits<double>::digits10 + 1) << min_dist
         << '\n'
         << hit_time_step << '\n'
         << gravity_device_id << ' ' << missile_cost << '\n';
}

void setup_gpu_constants() {
    double h_dt = param::dt;
    double h_eps = param::eps;
    double h_G = param::G;
    double h_planet_radius = param::planet_radius;
    double h_missile_speed = param::missile_speed;
    
    HIP_CHECK(hipMemcpyToSymbol(d_dt, &h_dt, sizeof(double)));
    HIP_CHECK(hipMemcpyToSymbol(d_eps, &h_eps, sizeof(double)));
    HIP_CHECK(hipMemcpyToSymbol(d_G, &h_G, sizeof(double)));
    HIP_CHECK(hipMemcpyToSymbol(d_planet_radius, &h_planet_radius, sizeof(double)));
    HIP_CHECK(hipMemcpyToSymbol(d_missile_speed, &h_missile_speed, sizeof(double)));
}

int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }

    Data ctx;
    read_input(argv[1], ctx);

    std::vector<double> saved_qx, saved_qy, saved_qz, saved_vx, saved_vy, saved_vz, saved_m;
    std::vector<int> saved_type;
    int saved_step = -1;
    
    saved_qx.resize(ctx.n);
    saved_qy.resize(ctx.n);
    saved_qz.resize(ctx.n);
    saved_vx.resize(ctx.n);
    saved_vy.resize(ctx.n);
    saved_vz.resize(ctx.n);
    saved_m.resize(ctx.n);
    saved_type.resize(ctx.n);

    // Problem 1 & 2
    double min_dist = std::numeric_limits<double>::infinity();
    int hit_time_step = -2;

    std::thread p1([&]() {
        HIP_CHECK(hipSetDevice(0));
        setup_gpu_constants();
        
        double *d_qx, *d_qy, *d_qz, *d_vx, *d_vy, *d_vz, *d_m;
        int *d_type;
        double *d_min_dist;
        
        HIP_CHECK(hipMalloc(&d_qx, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_qy, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_qz, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_vx, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_vy, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_vz, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_m, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_type, ctx.n * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_min_dist, sizeof(double)));

        // Prepare data for Prob 1 (devices have mass 0)
        std::vector<double> m_p1 = ctx.m;
        for(int i = 0; i < ctx.n; i++) {
            if(ctx.type[i] == 2) m_p1[i] = 0;
        }
        
        HIP_CHECK(hipMemcpy(d_qx, ctx.qx.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_qy, ctx.qy.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_qz, ctx.qz.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vx, ctx.vx.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vy, ctx.vy.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vz, ctx.vz.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_m, m_p1.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_type, ctx.type.data(), ctx.n * sizeof(int), hipMemcpyHostToDevice));
        
        double init_min_dist = std::numeric_limits<double>::infinity();
        HIP_CHECK(hipMemcpy(d_min_dist, &init_min_dist, sizeof(double), hipMemcpyHostToDevice));

        int blockSize = 256;
        int numBlocks = (ctx.n + blockSize - 1) / blockSize;
        
        for (int step = 0; step <= param::n_steps; step++) {
            if (step > 0) {
                compute_forces_update_velocity<<<numBlocks, blockSize>>>(ctx.n, d_qx, d_qy, d_qz, d_vx, d_vy, d_vz, d_m, nullptr, d_type, step * param::dt, ctx.planet, ctx.asteroid, d_min_dist);
                update_position<<<numBlocks, blockSize>>>(ctx.n, d_qx, d_qy, d_qz, d_vx, d_vy, d_vz);
            }
        }
        check_min_dist<<<1, 1>>>(ctx.planet, ctx.asteroid, d_qx, d_qy, d_qz, d_min_dist);
        
        HIP_CHECK(hipMemcpy(&min_dist, d_min_dist, sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipFree(d_qx)); HIP_CHECK(hipFree(d_qy)); HIP_CHECK(hipFree(d_qz));
        HIP_CHECK(hipFree(d_vx)); HIP_CHECK(hipFree(d_vy)); HIP_CHECK(hipFree(d_vz));
        HIP_CHECK(hipFree(d_m)); HIP_CHECK(hipFree(d_type)); HIP_CHECK(hipFree(d_min_dist));
    });

    std::thread p2([&]() {
        HIP_CHECK(hipSetDevice(1));
        setup_gpu_constants();
        
        double *d_qx, *d_qy, *d_qz, *d_vx, *d_vy, *d_vz, *d_m, *d_m0;
        int *d_type, *d_hit_step;
        
        double *d_saved_qx, *d_saved_qy, *d_saved_qz, *d_saved_vx, *d_saved_vy, *d_saved_vz, *d_saved_m;
        int *d_saved_type, *d_saved_step;
        
        HIP_CHECK(hipMalloc(&d_qx, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_qy, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_qz, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_vx, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_vy, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_vz, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_m, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_m0, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_type, ctx.n * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_hit_step, sizeof(int)));

        HIP_CHECK(hipMalloc(&d_saved_qx, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_saved_qy, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_saved_qz, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_saved_vx, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_saved_vy, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_saved_vz, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_saved_m, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_saved_type, ctx.n * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_saved_step, sizeof(int)));

        HIP_CHECK(hipMemcpy(d_qx, ctx.qx.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_qy, ctx.qy.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_qz, ctx.qz.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vx, ctx.vx.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vy, ctx.vy.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vz, ctx.vz.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_m, ctx.m.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_m0, ctx.m.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_type, ctx.type.data(), ctx.n * sizeof(int), hipMemcpyHostToDevice));
        
        int init_hit_step = -1;
        HIP_CHECK(hipMemcpy(d_hit_step, &init_hit_step, sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_saved_step, &init_hit_step, sizeof(int), hipMemcpyHostToDevice));
        
        int blockSize = 256;
        int numBlocks = (ctx.n + blockSize - 1) / blockSize;

        for (int step = 0; step <= param::n_steps; step++) {
            if (step > 0) {
                compute_forces_update_velocity<<<numBlocks, blockSize>>>(ctx.n, d_qx, d_qy, d_qz, d_vx, d_vy, d_vz, d_m, d_m0, d_type, step * param::dt, 0, 0, nullptr);
                update_position<<<numBlocks, blockSize>>>(ctx.n, d_qx, d_qy, d_qz, d_vx, d_vy, d_vz);
            }
            check_hits<<<numBlocks, blockSize>>>(ctx.n, ctx.planet, ctx.asteroid, d_qx, d_qy, d_qz, d_type, step, d_saved_step, d_hit_step);
            save_state<<<numBlocks, blockSize>>>(ctx.n, d_qx, d_qy, d_qz, d_vx, d_vy, d_vz, d_m, d_type, step, d_saved_step,
                d_saved_qx, d_saved_qy, d_saved_qz, d_saved_vx, d_saved_vy, d_saved_vz, d_saved_m, d_saved_type);
                    
            if (step % 2000 == 0) {
                int h_hit;
                HIP_CHECK(hipMemcpy(&h_hit, d_hit_step, sizeof(int), hipMemcpyDeviceToHost));
                if (h_hit != -1) break;
            }
        }
        
        int h_hit_step;
        HIP_CHECK(hipMemcpy(&h_hit_step, d_hit_step, sizeof(int), hipMemcpyDeviceToHost));
        
        if (h_hit_step != -1) {
            hit_time_step = h_hit_step;
        }

        HIP_CHECK(hipMemcpy(&saved_step, d_saved_step, sizeof(int), hipMemcpyDeviceToHost));
        if (saved_step != -1) {
            HIP_CHECK(hipMemcpy(saved_qx.data(), d_saved_qx, ctx.n * sizeof(double), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(saved_qy.data(), d_saved_qy, ctx.n * sizeof(double), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(saved_qz.data(), d_saved_qz, ctx.n * sizeof(double), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(saved_vx.data(), d_saved_vx, ctx.n * sizeof(double), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(saved_vy.data(), d_saved_vy, ctx.n * sizeof(double), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(saved_vz.data(), d_saved_vz, ctx.n * sizeof(double), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(saved_m.data(), d_saved_m, ctx.n * sizeof(double), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(saved_type.data(), d_saved_type, ctx.n * sizeof(int), hipMemcpyDeviceToHost));
        }
        
        HIP_CHECK(hipFree(d_qx)); HIP_CHECK(hipFree(d_qy)); HIP_CHECK(hipFree(d_qz));
        HIP_CHECK(hipFree(d_vx)); HIP_CHECK(hipFree(d_vy)); HIP_CHECK(hipFree(d_vz));
        HIP_CHECK(hipFree(d_m)); HIP_CHECK(hipFree(d_m0)); HIP_CHECK(hipFree(d_type)); HIP_CHECK(hipFree(d_hit_step));
        HIP_CHECK(hipFree(d_saved_qx)); HIP_CHECK(hipFree(d_saved_qy)); HIP_CHECK(hipFree(d_saved_qz));
        HIP_CHECK(hipFree(d_saved_vx)); HIP_CHECK(hipFree(d_saved_vy)); HIP_CHECK(hipFree(d_saved_vz));
        HIP_CHECK(hipFree(d_saved_m)); HIP_CHECK(hipFree(d_saved_type)); HIP_CHECK(hipFree(d_saved_step));
    });

    p1.join();
    p2.join();

    // Problem 3
    int gravity_device_id = -1;
    double missile_cost = 0.0;

    // Identify devices
    std::vector<int> devices;
    for(int i=0; i<ctx.n; ++i) {
        if(ctx.type[i] == 2) devices.push_back(i);
    }

    double best_cost = std::numeric_limits<double>::infinity();
    int best_id = -1;
    std::mutex result_mutex;

    auto worker = [&](int gpu_id, const std::vector<int>& device_subset) {
        HIP_CHECK(hipSetDevice(gpu_id));
        setup_gpu_constants();

        int blockSize = 256;
        int numBlocks = (ctx.n + blockSize - 1) / blockSize;

        double *d_qx, *d_qy, *d_qz, *d_vx, *d_vy, *d_vz, *d_m, *d_m0;
        int *d_type;
        int *d_hit_step, *d_destroyed_step;
        
        HIP_CHECK(hipMalloc(&d_qx, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_qy, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_qz, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_vx, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_vy, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_vz, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_m, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_m0, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_type, ctx.n * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_hit_step, sizeof(int)));
        HIP_CHECK(hipMalloc(&d_destroyed_step, sizeof(int)));

        HIP_CHECK(hipMemcpy(d_m0, ctx.m.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));

        for (int d_idx : device_subset) {
            int start_step = 0;
            if (saved_step != -1) {
                start_step = saved_step;
                HIP_CHECK(hipMemcpy(d_qx, saved_qx.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(d_qy, saved_qy.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(d_qz, saved_qz.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(d_vx, saved_vx.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(d_vy, saved_vy.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(d_vz, saved_vz.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(d_m, saved_m.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(d_type, saved_type.data(), ctx.n * sizeof(int), hipMemcpyHostToDevice));
            } else {
                HIP_CHECK(hipMemcpy(d_qx, ctx.qx.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(d_qy, ctx.qy.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(d_qz, ctx.qz.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(d_vx, ctx.vx.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(d_vy, ctx.vy.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(d_vz, ctx.vz.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(d_m, ctx.m.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(d_type, ctx.type.data(), ctx.n * sizeof(int), hipMemcpyHostToDevice));
            }
            
            int init_val = -1;
            HIP_CHECK(hipMemcpy(d_hit_step, &init_val, sizeof(int), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_destroyed_step, &init_val, sizeof(int), hipMemcpyHostToDevice));

            for (int step = start_step; step <= param::n_steps; step++) {
                if (step > start_step) {
                    compute_forces_update_velocity<<<numBlocks, blockSize>>>(ctx.n, d_qx, d_qy, d_qz, d_vx, d_vy, d_vz, d_m, d_m0, d_type, step * param::dt, 0, 0, nullptr);
                    update_position<<<numBlocks, blockSize>>>(ctx.n, d_qx, d_qy, d_qz, d_vx, d_vy, d_vz);
                }
                check_hit_and_destroy<<<1, 1>>>(ctx.planet, ctx.asteroid, d_idx, d_qx, d_qy, d_qz, d_m, d_type, step, d_hit_step, d_destroyed_step);
                                
                if (step % 2000 == 0) {
                    int h_hit;
                    HIP_CHECK(hipMemcpy(&h_hit, d_hit_step, sizeof(int), hipMemcpyDeviceToHost));
                    if (h_hit != -1) break;
                }
            }
            
            int h_hit_step, h_destroyed_step;
            HIP_CHECK(hipMemcpy(&h_hit_step, d_hit_step, sizeof(int), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(&h_destroyed_step, d_destroyed_step, sizeof(int), hipMemcpyDeviceToHost));
            
            if (h_hit_step == -1) {
                double current_cost = 0.0;
                if (h_destroyed_step != -1) {
                    current_cost = param::get_missile_cost(h_destroyed_step * param::dt);
                }
                
                std::lock_guard<std::mutex> lock(result_mutex);
                if (current_cost < best_cost) {
                    best_cost = current_cost;
                    best_id = d_idx;
                }
            }
        }
        
        HIP_CHECK(hipFree(d_qx)); HIP_CHECK(hipFree(d_qy)); HIP_CHECK(hipFree(d_qz));
        HIP_CHECK(hipFree(d_vx)); HIP_CHECK(hipFree(d_vy)); HIP_CHECK(hipFree(d_vz));
        HIP_CHECK(hipFree(d_m)); HIP_CHECK(hipFree(d_m0)); HIP_CHECK(hipFree(d_type));
        HIP_CHECK(hipFree(d_hit_step)); HIP_CHECK(hipFree(d_destroyed_step));
    };

    std::vector<int> devices0, devices1;
    for (int i = 0; i < devices.size(); ++i) {
        if (i & 1) devices1.push_back(devices[i]);
        else devices0.push_back(devices[i]);
    }

    std::thread p3_0(worker, 0, devices0);
    std::thread p3_1(worker, 1, devices1);
    p3_0.join();
    p3_1.join();

    if (best_id != -1) {
        gravity_device_id = best_id;
        missile_cost = best_cost;
    }

    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);
    
    return 0;
}