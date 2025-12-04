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
#include <algorithm>
#include <atomic>
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
__constant__ double d_m0[1024];

__global__ void update_mass(int n, double* m, int* type, double t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && type[i] == 2) {
        double tmp = d_m0[i];
        m[i] = tmp + 0.5 * tmp * fabs(sin(t / 6000.0));
    }
}

__global__ void check_min_dist(int planet, int asteroid, double* qx, double* qy, double* qz, double* min_dist) {
    if (threadIdx.x == 0) {
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        double dist = sqrt(dx * dx + dy * dy + dz * dz);
        atomicMin(min_dist, dist);
    }
}

__global__ void compute_forces_and_integrate(int n, const double* in_qx, const double* in_qy, const double* in_qz,
                                              const double* m, double* vx, double* vy, double* vz,
                                              double* out_qx, double* out_qy, double* out_qz,
                                              int planet, int asteroid, double* min_dist) {
    extern __shared__ double shared_mem[];
    double* sx = shared_mem;
    double* sy = shared_mem + n;
    double* sz = shared_mem + 2 * n;

    int i = blockIdx.x;
    int j = threadIdx.x;

    double in_qx_i = in_qx[i];
    double in_qy_i = in_qy[i];
    double in_qz_i = in_qz[i];

    double dx = in_qx[j] - in_qx_i;
    double dy = in_qy[j] - in_qy_i;
    double dz = in_qz[j] - in_qz_i;
    double dist2 = dx * dx + dy * dy + dz * dz + d_eps * d_eps;
    double invDist = rsqrt(dist2);
    double invDist3 = invDist * invDist * invDist;
    double common = d_G * m[j] * invDist3;

    sx[j] = common * dx;
    sy[j] = common * dy;
    sz[j] = common * dz;

    __syncthreads();

    // Shared memory reduction until we have 64 or fewer elements
    unsigned int len = blockDim.x;
    while (len > 64) {
        unsigned int stride = (len + 1) / 2;
        if (j < len / 2) {
            sx[j] += sx[j + stride];
            sy[j] += sy[j + stride];
            sz[j] += sz[j + stride];
        }
        len = stride;
        __syncthreads();
    }

    // Warp-level reduction for last 64 elements (warp size on AMD)
    if (j < 64) {
        double val_x = sx[j];
        double val_y = sy[j];
        double val_z = sz[j];
        
        // Reduce within warp using shuffle - no sync needed
        for (int offset = 32; offset > 0; offset /= 2) {
            val_x += __shfl_down(val_x, offset);
            val_y += __shfl_down(val_y, offset);
            val_z += __shfl_down(val_z, offset);
        }
        
        // Thread 0 has the final sum
        if (j == 0) {
            double new_vx = vx[i] + val_x * d_dt;
            double new_vy = vy[i] + val_y * d_dt;
            double new_vz = vz[i] + val_z * d_dt;
            
            vx[i] = new_vx;
            vy[i] = new_vy;
            vz[i] = new_vz;
            
            out_qx[i] = in_qx_i + new_vx * d_dt;
            out_qy[i] = in_qy_i + new_vy * d_dt;
            out_qz[i] = in_qz_i + new_vz * d_dt;
            
            if (i == 0 && min_dist != nullptr) {
                double pdx = in_qx[planet] - in_qx[asteroid];
                double pdy = in_qy[planet] - in_qy[asteroid];
                double pdz = in_qz[planet] - in_qz[asteroid];
                double dist = sqrt(pdx * pdx + pdy * pdy + pdz * pdz);
                atomicMin(min_dist, dist);
            }
        }
    }
}

__global__ void check_hits_and_reachable(int n, int planet, int asteroid, double* qx, double* qy, double* qz, 
                                         int* type, int step, int* hit_step,
                                         int* device_reached, double* device_dist) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check planet-asteroid collision (only thread 0)
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

    // Check device reachability
    if (type[i] == 2) {
        if (device_reached[i] != -1) return;

        double dx = qx[planet] - qx[i];
        double dy = qy[planet] - qy[i];
        double dz = qz[planet] - qz[i];
        double dist = sqrt(dx * dx + dy * dy + dz * dz);
        
        double missile_dist = step * d_dt * d_missile_speed;
        if (missile_dist > dist) {
            device_reached[i] = step;
            device_dist[i] = dist;
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

struct DeviceState {
    int device_id;
    int step;
    double distance;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<int> type;
};

void read_input(const char* filename, Data& ctx) {
    std::ifstream fin(filename);
    fin >> ctx.n >> ctx.planet >> ctx.asteroid;
    ctx.qx.resize(ctx.n); ctx.qy.resize(ctx.n); ctx.qz.resize(ctx.n);
    ctx.vx.resize(ctx.n); ctx.vy.resize(ctx.n); ctx.vz.resize(ctx.n);
    ctx.m.resize(ctx.n); ctx.type.resize(ctx.n);
    
    for (int i = 0; i < ctx.n; i++) {
        std::string t;
        fin >> ctx.qx[i] >> ctx.qy[i] >> ctx.qz[i] 
            >> ctx.vx[i] >> ctx.vy[i] >> ctx.vz[i] 
            >> ctx.m[i] >> t;
        if (t == "planet") ctx.type[i] = 0;
        else if (t == "asteroid") ctx.type[i] = 1;
        else if (t == "device") ctx.type[i] = 2;
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

void setup_gpu_constants(const std::vector<double>& m0) {
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
    HIP_CHECK(hipMemcpyToSymbol(d_m0, m0.data(), m0.size() * sizeof(double)));
}

int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }

    Data ctx;
    read_input(argv[1], ctx);

    std::vector<DeviceState> device_states;
    std::mutex device_states_mutex;

    // Problem 1 & 2
    double min_dist = std::numeric_limits<double>::infinity();
    int hit_time_step = -2;

    std::thread p1([&]() {
        HIP_CHECK(hipSetDevice(0));
        setup_gpu_constants(ctx.m);
        
        double *d_qx[2], *d_qy[2], *d_qz[2];
        double *d_vx, *d_vy, *d_vz;
        double *d_m;
        int *d_type;
        double *d_min_dist;
        
        for(int k = 0; k < 2; k++) {
            HIP_CHECK(hipMalloc(&d_qx[k], ctx.n * sizeof(double)));
            HIP_CHECK(hipMalloc(&d_qy[k], ctx.n * sizeof(double)));
            HIP_CHECK(hipMalloc(&d_qz[k], ctx.n * sizeof(double)));
        }
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
        
        HIP_CHECK(hipMemcpy(d_qx[0], ctx.qx.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_qy[0], ctx.qy.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_qz[0], ctx.qz.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vx, ctx.vx.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vy, ctx.vy.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vz, ctx.vz.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_m, m_p1.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_type, ctx.type.data(), ctx.n * sizeof(int), hipMemcpyHostToDevice));
        
        double init_min_dist = std::numeric_limits<double>::infinity();
        HIP_CHECK(hipMemcpy(d_min_dist, &init_min_dist, sizeof(double), hipMemcpyHostToDevice));

        int blockSize = 1024;
        int numBlocks = (ctx.n + blockSize - 1) / blockSize;
        
        int n_steps = param::n_steps;
        double dt = param::dt;
        
        int in = 0;
        int out = 1;
        size_t shared_mem_size = 3 * ctx.n * sizeof(double);
        for (int step = 1; step <= n_steps; step++) {
            compute_forces_and_integrate<<<ctx.n, ctx.n, shared_mem_size>>>(ctx.n,
                d_qx[in], d_qy[in], d_qz[in],
                d_m, d_vx, d_vy, d_vz,
                d_qx[out], d_qy[out], d_qz[out],
                ctx.planet, ctx.asteroid, d_min_dist);
            std::swap(in, out);
        }
        check_min_dist<<<1, 1>>>(ctx.planet, ctx.asteroid, d_qx[in], d_qy[in], d_qz[in], d_min_dist);
    
        HIP_CHECK(hipMemcpy(&min_dist, d_min_dist, sizeof(double), hipMemcpyDeviceToHost));
        for(int k = 0; k < 2; k++) {
            HIP_CHECK(hipFree(d_qx[k])); HIP_CHECK(hipFree(d_qy[k])); HIP_CHECK(hipFree(d_qz[k]));
        }
        HIP_CHECK(hipFree(d_vx)); HIP_CHECK(hipFree(d_vy)); HIP_CHECK(hipFree(d_vz));
        HIP_CHECK(hipFree(d_m));
        HIP_CHECK(hipFree(d_type)); HIP_CHECK(hipFree(d_min_dist));
    });

    std::thread p2([&]() {
        HIP_CHECK(hipSetDevice(1));
        setup_gpu_constants(ctx.m);
        
        double *d_qx[2], *d_qy[2], *d_qz[2];
        double *d_vx, *d_vy, *d_vz;
        double *d_m;
        int *d_type, *d_hit_step;
        int *d_device_reached;
        double *d_device_dist;
        
        for(int k = 0; k < 2; k++) {
            HIP_CHECK(hipMalloc(&d_qx[k], ctx.n * sizeof(double)));
            HIP_CHECK(hipMalloc(&d_qy[k], ctx.n * sizeof(double)));
            HIP_CHECK(hipMalloc(&d_qz[k], ctx.n * sizeof(double)));
        }
        HIP_CHECK(hipMalloc(&d_vx, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_vy, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_vz, ctx.n * sizeof(double)));

        HIP_CHECK(hipMalloc(&d_m, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_type, ctx.n * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_hit_step, sizeof(int)));
        HIP_CHECK(hipMalloc(&d_device_reached, ctx.n * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_device_dist, ctx.n * sizeof(double)));

        HIP_CHECK(hipMemcpy(d_qx[0], ctx.qx.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_qy[0], ctx.qy.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_qz[0], ctx.qz.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vx, ctx.vx.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vy, ctx.vy.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vz, ctx.vz.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_m, ctx.m.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_type, ctx.type.data(), ctx.n * sizeof(int), hipMemcpyHostToDevice));
        
        int init_hit_step = -1;
        HIP_CHECK(hipMemcpy(d_hit_step, &init_hit_step, sizeof(int), hipMemcpyHostToDevice));
        
        // Initialize device_reached to -1 for all
        std::vector<int> h_device_reached(ctx.n, -1);
        std::vector<double> h_device_dist(ctx.n, std::numeric_limits<double>::infinity());
        HIP_CHECK(hipMemcpy(d_device_reached, h_device_reached.data(), ctx.n * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_device_dist, h_device_dist.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
        
        // Track which devices we've saved state for
        std::vector<bool> device_saved(ctx.n, false);
        
        int blockSize = 1024;
        int numBlocks = (ctx.n + blockSize - 1) / blockSize;

        int in = 0;
        int out = 1;

        // Step 0
        check_hits_and_reachable<<<1, ctx.n>>>(ctx.n, ctx.planet, ctx.asteroid, d_qx[in], d_qy[in], d_qz[in], d_type, 0, d_hit_step, d_device_reached, d_device_dist);

        size_t shared_mem_size = 3 * ctx.n * sizeof(double);
        for (int step = 1; step <= param::n_steps; step++) {
            update_mass<<<numBlocks, blockSize>>>(ctx.n, d_m, d_type, step * param::dt);

            compute_forces_and_integrate<<<ctx.n, ctx.n, shared_mem_size>>>(ctx.n,
                d_qx[in], d_qy[in], d_qz[in],
                d_m, d_vx, d_vy, d_vz,
                d_qx[out], d_qy[out], d_qz[out],
                ctx.planet, ctx.asteroid, nullptr);

            check_hits_and_reachable<<<1, ctx.n>>>(ctx.n, ctx.planet, ctx.asteroid, d_qx[out], d_qy[out], d_qz[out], d_type, step, d_hit_step, d_device_reached, d_device_dist);

            std::swap(in, out);

            // Check periodically for newly reachable devices and save their states
            if (step % 100 == 0) {
                HIP_CHECK(hipMemcpy(h_device_reached.data(), d_device_reached, ctx.n * sizeof(int), hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(h_device_dist.data(), d_device_dist, ctx.n * sizeof(double), hipMemcpyDeviceToHost));
                
                for (int i = 0; i < ctx.n; i++) {
                    if (ctx.type[i] == 2 && h_device_reached[i] != -1 && !device_saved[i]) {
                        // Save state for this device
                        DeviceState ds;
                        ds.device_id = i;
                        ds.step = h_device_reached[i];
                        ds.distance = h_device_dist[i];
                        ds.qx.resize(ctx.n); ds.qy.resize(ctx.n); ds.qz.resize(ctx.n);
                        ds.vx.resize(ctx.n); ds.vy.resize(ctx.n); ds.vz.resize(ctx.n);
                        ds.m.resize(ctx.n); ds.type.resize(ctx.n);
                        
                        HIP_CHECK(hipMemcpy(ds.qx.data(), d_qx[in], ctx.n * sizeof(double), hipMemcpyDeviceToHost));
                        HIP_CHECK(hipMemcpy(ds.qy.data(), d_qy[in], ctx.n * sizeof(double), hipMemcpyDeviceToHost));
                        HIP_CHECK(hipMemcpy(ds.qz.data(), d_qz[in], ctx.n * sizeof(double), hipMemcpyDeviceToHost));
                        HIP_CHECK(hipMemcpy(ds.vx.data(), d_vx, ctx.n * sizeof(double), hipMemcpyDeviceToHost));
                        HIP_CHECK(hipMemcpy(ds.vy.data(), d_vy, ctx.n * sizeof(double), hipMemcpyDeviceToHost));
                        HIP_CHECK(hipMemcpy(ds.vz.data(), d_vz, ctx.n * sizeof(double), hipMemcpyDeviceToHost));
                        HIP_CHECK(hipMemcpy(ds.m.data(), d_m, ctx.n * sizeof(double), hipMemcpyDeviceToHost));
                        HIP_CHECK(hipMemcpy(ds.type.data(), d_type, ctx.n * sizeof(int), hipMemcpyDeviceToHost));
                        
                        {
                            std::lock_guard<std::mutex> lock(device_states_mutex);
                            device_states.push_back(std::move(ds));
                        }
                        device_saved[i] = true;
                    }
                }
                
                int h_hit_step;
                HIP_CHECK(hipMemcpy(&h_hit_step, d_hit_step, sizeof(int), hipMemcpyDeviceToHost));
                if (h_hit_step != -1) break;
            }
        }
        
        // Final check for any remaining devices
        HIP_CHECK(hipMemcpy(h_device_reached.data(), d_device_reached, ctx.n * sizeof(int), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_device_dist.data(), d_device_dist, ctx.n * sizeof(double), hipMemcpyDeviceToHost));
        
        for (int i = 0; i < ctx.n; i++) {
            if (ctx.type[i] == 2 && h_device_reached[i] != -1 && !device_saved[i]) {
                DeviceState ds;
                ds.device_id = i;
                ds.step = h_device_reached[i];
                ds.distance = h_device_dist[i];
                ds.qx.resize(ctx.n); ds.qy.resize(ctx.n); ds.qz.resize(ctx.n);
                ds.vx.resize(ctx.n); ds.vy.resize(ctx.n); ds.vz.resize(ctx.n);
                ds.m.resize(ctx.n); ds.type.resize(ctx.n);
                
                HIP_CHECK(hipMemcpy(ds.qx.data(), d_qx[in], ctx.n * sizeof(double), hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(ds.qy.data(), d_qy[in], ctx.n * sizeof(double), hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(ds.qz.data(), d_qz[in], ctx.n * sizeof(double), hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(ds.vx.data(), d_vx, ctx.n * sizeof(double), hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(ds.vy.data(), d_vy, ctx.n * sizeof(double), hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(ds.vz.data(), d_vz, ctx.n * sizeof(double), hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(ds.m.data(), d_m, ctx.n * sizeof(double), hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(ds.type.data(), d_type, ctx.n * sizeof(int), hipMemcpyDeviceToHost));
                
                {
                    std::lock_guard<std::mutex> lock(device_states_mutex);
                    device_states.push_back(std::move(ds));
                }
                device_saved[i] = true;
            }
        }
        
        int h_hit_step;
        HIP_CHECK(hipMemcpy(&h_hit_step, d_hit_step, sizeof(int), hipMemcpyDeviceToHost));
        
        if (h_hit_step != -1) {
            hit_time_step = h_hit_step;
        }
        
        for(int k = 0; k < 2; k++) {
            HIP_CHECK(hipFree(d_qx[k])); HIP_CHECK(hipFree(d_qy[k])); HIP_CHECK(hipFree(d_qz[k]));
        }
        HIP_CHECK(hipFree(d_vx)); HIP_CHECK(hipFree(d_vy)); HIP_CHECK(hipFree(d_vz));
        HIP_CHECK(hipFree(d_m));
        HIP_CHECK(hipFree(d_type)); HIP_CHECK(hipFree(d_hit_step));
        HIP_CHECK(hipFree(d_device_reached)); HIP_CHECK(hipFree(d_device_dist));
    });

    p1.join();
    p2.join();

    // Problem 3
    int gravity_device_id = -1;
    double missile_cost = 0.0;

    // Sort device_states by distance (smaller distance = lower missile cost)
    std::sort(device_states.begin(), device_states.end(), [](const DeviceState& a, const DeviceState& b) {
        return a.distance < b.distance;
    });

    std::atomic<bool> solution_found(false);
    double best_cost = std::numeric_limits<double>::infinity();
    int best_id = -1;
    std::mutex result_mutex;

    auto worker = [&](int gpu_id, const std::vector<size_t>& state_indices) {
        HIP_CHECK(hipSetDevice(gpu_id));
        setup_gpu_constants(ctx.m);

        int blockSize = 1024;
        int numBlocks = (ctx.n + blockSize - 1) / blockSize;

        double *d_qx[2], *d_qy[2], *d_qz[2];
        double *d_vx, *d_vy, *d_vz;
        double *d_m;
        int *d_type;
        int *d_hit_step, *d_destroyed_step;
        
        for(int k = 0; k < 2; k++) {
            HIP_CHECK(hipMalloc(&d_qx[k], ctx.n * sizeof(double)));
            HIP_CHECK(hipMalloc(&d_qy[k], ctx.n * sizeof(double)));
            HIP_CHECK(hipMalloc(&d_qz[k], ctx.n * sizeof(double)));
        }
        HIP_CHECK(hipMalloc(&d_vx, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_vy, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_vz, ctx.n * sizeof(double)));

        HIP_CHECK(hipMalloc(&d_m, ctx.n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_type, ctx.n * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_hit_step, sizeof(int)));
        HIP_CHECK(hipMalloc(&d_destroyed_step, sizeof(int)));

        for (size_t idx : state_indices) {
            // Early return if solution already found
            if (solution_found.load()) break;
            
            const DeviceState& ds = device_states[idx];
            int d_idx = ds.device_id;
            int start_step = ds.step;
            
            HIP_CHECK(hipMemcpy(d_qx[0], ds.qx.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_qy[0], ds.qy.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_qz[0], ds.qz.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_vx, ds.vx.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_vy, ds.vy.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_vz, ds.vz.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_m, ds.m.data(), ctx.n * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_type, ds.type.data(), ctx.n * sizeof(int), hipMemcpyHostToDevice));
            
            int init_val = -1;
            HIP_CHECK(hipMemcpy(d_hit_step, &init_val, sizeof(int), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_destroyed_step, &init_val, sizeof(int), hipMemcpyHostToDevice));

            int in = 0;
            int out = 1;
            int blockSize = 1024;
            int numBlocks = (ctx.n + blockSize - 1) / blockSize;

            // Step start_step
            check_hit_and_destroy<<<1, 1>>>(ctx.planet, ctx.asteroid, d_idx, d_qx[in], d_qy[in], d_qz[in], d_m, d_type, start_step, d_hit_step, d_destroyed_step);

            size_t shared_mem_size = 3 * ctx.n * sizeof(double);
            for (int step = start_step + 1; step <= param::n_steps; step++) {
                update_mass<<<numBlocks, blockSize>>>(ctx.n, d_m, d_type, step * param::dt);

                compute_forces_and_integrate<<<ctx.n, ctx.n, shared_mem_size>>>(ctx.n,
                    d_qx[in], d_qy[in], d_qz[in],
                    d_m, d_vx, d_vy, d_vz,
                    d_qx[out], d_qy[out], d_qz[out],
                    ctx.planet, ctx.asteroid, nullptr);

                check_hit_and_destroy<<<1, 1>>>(ctx.planet, ctx.asteroid, d_idx, d_qx[out], d_qy[out], d_qz[out], d_m, d_type, step, d_hit_step, d_destroyed_step);

                std::swap(in, out);

                if (step % 2000 == 0) {
                    int h_hit_step;
                    HIP_CHECK(hipMemcpy(&h_hit_step, d_hit_step, sizeof(int), hipMemcpyDeviceToHost));
                    if (h_hit_step != -1) break;
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
                // Signal that we found a solution (since devices are sorted by distance,
                // the first successful one has the lowest cost)
                solution_found.store(true);
                break;
            }
        }
        
        for(int k = 0; k < 2; k++) {
            HIP_CHECK(hipFree(d_qx[k])); HIP_CHECK(hipFree(d_qy[k])); HIP_CHECK(hipFree(d_qz[k]));
        }
        HIP_CHECK(hipFree(d_vx)); HIP_CHECK(hipFree(d_vy)); HIP_CHECK(hipFree(d_vz));
        HIP_CHECK(hipFree(d_m));
        HIP_CHECK(hipFree(d_type));
        HIP_CHECK(hipFree(d_hit_step)); HIP_CHECK(hipFree(d_destroyed_step));
    };

    // Distribute sorted devices between GPUs (alternating for load balance)
    std::vector<size_t> indices0, indices1;
    for (size_t i = 0; i < device_states.size(); ++i) {
        if (i & 1) indices1.push_back(i);
        else indices0.push_back(i);
    }

    std::thread p3_0(worker, 0, indices0);
    std::thread p3_1(worker, 1, indices1);
    p3_0.join();
    p3_1.join();

    if (best_id != -1) {
        gravity_device_id = best_id;
        missile_cost = best_cost;
    }

    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);
    
    return 0;
}