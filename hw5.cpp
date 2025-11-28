#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <hip/hip_runtime.h>

#define CHECK(cmd) \
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
    double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
}

// Device constants
__constant__ double d_dt;
__constant__ double d_eps;
__constant__ double d_G;
__constant__ double d_planet_radius;
__constant__ double d_missile_speed;

__device__ double gravity_device_mass(double m0, double t) {
    return m0 + 0.5 * m0 * fabs(sin(t / 6000.0));
}

__global__ void check_min_dist_kernel(int planet, int asteroid, double* qx, double* qy, double* qz, double* min_dist) {
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

__global__ void compute_forces_update_v(int n, double* qx, double* qy, double* qz,
                                        double* vx, double* vy, double* vz,
                                        double* m, int* type, double t,
                                        int planet, int asteroid, double* min_dist) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (min_dist != nullptr && blockIdx.x == 0 && threadIdx.x == 0) {
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        double dist = sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < *min_dist) {
            *min_dist = dist;
        }
    }

    if (i >= n) return;

    double ax = 0.0, ay = 0.0, az = 0.0;
    double my_qx = qx[i];
    double my_qy = qy[i];
    double my_qz = qz[i];

    for (int j = 0; j < n; j++) {
        if (i == j) continue;
        double mj = m[j];
        // type: 0=planet, 1=asteroid, 2=device, 3=destroyed
        if (type[j] == 2) {
            mj = gravity_device_mass(mj, t);
        }
        
        double dx = qx[j] - my_qx;
        double dy = qy[j] - my_qy;
        double dz = qz[j] - my_qz;
        double dist2 = dx * dx + dy * dy + dz * dz + d_eps * d_eps;
        double dist3 = dist2 * sqrt(dist2);
        
        double f = d_G * mj / dist3;
        ax += f * dx;
        ay += f * dy;
        az += f * dz;
    }

    vx[i] += ax * d_dt;
    vy[i] += ay * d_dt;
    vz[i] += az * d_dt;
}

__global__ void update_positions(int n, double* qx, double* qy, double* qz,
                                 double* vx, double* vy, double* vz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;
    qx[i] += vx[i] * d_dt;
    qy[i] += vy[i] * d_dt;
    qz[i] += vz[i] * d_dt;
}

__global__ void check_collision_kernel(int planet, int asteroid, double* qx, double* qy, double* qz, int* hit_step, int step) {
    if (threadIdx.x == 0) {
        // Only update if not already hit
        if (*hit_step == -1) {
            double dx = qx[planet] - qx[asteroid];
            double dy = qy[planet] - qy[asteroid];
            double dz = qz[planet] - qz[asteroid];
            if (dx * dx + dy * dy + dz * dz < d_planet_radius * d_planet_radius) {
                *hit_step = step;
            }
        }
    }
}

__global__ void check_missile_kernel(int planet, int target_device, double* qx, double* qy, double* qz, 
                                     double* m, int* type, int step, int* destroyed_step) {
    if (threadIdx.x == 0 && *destroyed_step == -1) {
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

struct SystemData {
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<int> type;
    int n, planet, asteroid;
};

void read_input(const char* filename, SystemData& sys) {
    std::ifstream fin(filename);
    fin >> sys.n >> sys.planet >> sys.asteroid;
    sys.qx.resize(sys.n); sys.qy.resize(sys.n); sys.qz.resize(sys.n);
    sys.vx.resize(sys.n); sys.vy.resize(sys.n); sys.vz.resize(sys.n);
    sys.m.resize(sys.n); sys.type.resize(sys.n);
    
    for (int i = 0; i < sys.n; i++) {
        std::string t;
        fin >> sys.qx[i] >> sys.qy[i] >> sys.qz[i] 
            >> sys.vx[i] >> sys.vy[i] >> sys.vz[i] 
            >> sys.m[i] >> t;
        if (t == "planet") sys.type[i] = 0;
        else if (t == "asteroid") sys.type[i] = 1;
        else if (t == "device") sys.type[i] = 2;
        else sys.type[i] = 3;
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

// Helper to setup constants on GPU
void setup_gpu_constants() {
    double h_dt = param::dt;
    double h_eps = param::eps;
    double h_G = param::G;
    double h_planet_radius = param::planet_radius;
    double h_missile_speed = param::missile_speed;
    
    CHECK(hipMemcpyToSymbol(d_dt, &h_dt, sizeof(double)));
    CHECK(hipMemcpyToSymbol(d_eps, &h_eps, sizeof(double)));
    CHECK(hipMemcpyToSymbol(d_G, &h_G, sizeof(double)));
    CHECK(hipMemcpyToSymbol(d_planet_radius, &h_planet_radius, sizeof(double)));
    CHECK(hipMemcpyToSymbol(d_missile_speed, &h_missile_speed, sizeof(double)));
}

int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }

    SystemData sys;
    read_input(argv[1], sys);

    // Register host memory for async copy
    CHECK(hipHostRegister(sys.qx.data(), sys.n * sizeof(double), hipHostRegisterDefault));
    CHECK(hipHostRegister(sys.qy.data(), sys.n * sizeof(double), hipHostRegisterDefault));
    CHECK(hipHostRegister(sys.qz.data(), sys.n * sizeof(double), hipHostRegisterDefault));
    CHECK(hipHostRegister(sys.vx.data(), sys.n * sizeof(double), hipHostRegisterDefault));
    CHECK(hipHostRegister(sys.vy.data(), sys.n * sizeof(double), hipHostRegisterDefault));
    CHECK(hipHostRegister(sys.vz.data(), sys.n * sizeof(double), hipHostRegisterDefault));
    CHECK(hipHostRegister(sys.m.data(), sys.n * sizeof(double), hipHostRegisterDefault));
    CHECK(hipHostRegister(sys.type.data(), sys.n * sizeof(int), hipHostRegisterDefault));

    // Problem 1 & 2
    
    double min_dist = std::numeric_limits<double>::infinity();
    int hit_time_step = -2;
    
    auto p1_start = std::chrono::high_resolution_clock::now();

    std::thread t1([&]() {
        CHECK(hipSetDevice(0));
        setup_gpu_constants();
        
        hipStream_t stream;
        CHECK(hipStreamCreate(&stream));

        double *d_qx, *d_qy, *d_qz, *d_vx, *d_vy, *d_vz, *d_m;
        int *d_type;
        double *d_min_dist;
        
        CHECK(hipMalloc(&d_qx, sys.n * sizeof(double)));
        CHECK(hipMalloc(&d_qy, sys.n * sizeof(double)));
        CHECK(hipMalloc(&d_qz, sys.n * sizeof(double)));
        CHECK(hipMalloc(&d_vx, sys.n * sizeof(double)));
        CHECK(hipMalloc(&d_vy, sys.n * sizeof(double)));
        CHECK(hipMalloc(&d_vz, sys.n * sizeof(double)));
        CHECK(hipMalloc(&d_m, sys.n * sizeof(double)));
        CHECK(hipMalloc(&d_type, sys.n * sizeof(int)));
        CHECK(hipMalloc(&d_min_dist, sizeof(double)));

        // Prepare data for Prob 1 (devices have mass 0)
        std::vector<double> m_p1 = sys.m;
        for(int i=0; i<sys.n; ++i) {
            if(sys.type[i] == 2) m_p1[i] = 0;
        }
        
        // Register m_p1 for async copy
        CHECK(hipHostRegister(m_p1.data(), sys.n * sizeof(double), hipHostRegisterDefault));

        CHECK(hipMemcpyAsync(d_qx, sys.qx.data(), sys.n * sizeof(double), hipMemcpyHostToDevice, stream));
        CHECK(hipMemcpyAsync(d_qy, sys.qy.data(), sys.n * sizeof(double), hipMemcpyHostToDevice, stream));
        CHECK(hipMemcpyAsync(d_qz, sys.qz.data(), sys.n * sizeof(double), hipMemcpyHostToDevice, stream));
        CHECK(hipMemcpyAsync(d_vx, sys.vx.data(), sys.n * sizeof(double), hipMemcpyHostToDevice, stream));
        CHECK(hipMemcpyAsync(d_vy, sys.vy.data(), sys.n * sizeof(double), hipMemcpyHostToDevice, stream));
        CHECK(hipMemcpyAsync(d_vz, sys.vz.data(), sys.n * sizeof(double), hipMemcpyHostToDevice, stream));
        CHECK(hipMemcpyAsync(d_m, m_p1.data(), sys.n * sizeof(double), hipMemcpyHostToDevice, stream));
        CHECK(hipMemcpyAsync(d_type, sys.type.data(), sys.n * sizeof(int), hipMemcpyHostToDevice, stream));
        
        double init_min_dist = std::numeric_limits<double>::infinity();
        CHECK(hipMemcpyAsync(d_min_dist, &init_min_dist, sizeof(double), hipMemcpyHostToDevice, stream));

        int blockSize = 256;
        int numBlocks = (sys.n + blockSize - 1) / blockSize;
        
        for (int step = 0; step <= param::n_steps; step++) {
            if (step > 0) {
                compute_forces_update_v<<<numBlocks, blockSize, 0, stream>>>(sys.n, d_qx, d_qy, d_qz, d_vx, d_vy, d_vz, d_m, d_type, step * param::dt, sys.planet, sys.asteroid, d_min_dist);
                update_positions<<<numBlocks, blockSize, 0, stream>>>(sys.n, d_qx, d_qy, d_qz, d_vx, d_vy, d_vz);
            }
        }
        
        // Check final position
        check_min_dist_kernel<<<1, 1, 0, stream>>>(sys.planet, sys.asteroid, d_qx, d_qy, d_qz, d_min_dist);
        
        CHECK(hipMemcpyAsync(&min_dist, d_min_dist, sizeof(double), hipMemcpyDeviceToHost, stream));
        CHECK(hipStreamSynchronize(stream));
        
        CHECK(hipHostUnregister(m_p1.data()));
        CHECK(hipFree(d_qx)); CHECK(hipFree(d_qy)); CHECK(hipFree(d_qz));
        CHECK(hipFree(d_vx)); CHECK(hipFree(d_vy)); CHECK(hipFree(d_vz));
        CHECK(hipFree(d_m)); CHECK(hipFree(d_type)); CHECK(hipFree(d_min_dist));
        CHECK(hipStreamDestroy(stream));
    });

    std::thread t2([&]() {
        CHECK(hipSetDevice(1));
        setup_gpu_constants();
        
        hipStream_t stream;
        CHECK(hipStreamCreate(&stream));
        
        double *d_qx, *d_qy, *d_qz, *d_vx, *d_vy, *d_vz, *d_m;
        int *d_type;
        int *d_hit_step;
        double *d_dummy_dist; // For update_positions
        
        CHECK(hipMalloc(&d_qx, sys.n * sizeof(double)));
        CHECK(hipMalloc(&d_qy, sys.n * sizeof(double)));
        CHECK(hipMalloc(&d_qz, sys.n * sizeof(double)));
        CHECK(hipMalloc(&d_vx, sys.n * sizeof(double)));
        CHECK(hipMalloc(&d_vy, sys.n * sizeof(double)));
        CHECK(hipMalloc(&d_vz, sys.n * sizeof(double)));
        CHECK(hipMalloc(&d_m, sys.n * sizeof(double)));
        CHECK(hipMalloc(&d_type, sys.n * sizeof(int)));
        CHECK(hipMalloc(&d_hit_step, sizeof(int)));
        CHECK(hipMalloc(&d_dummy_dist, sizeof(double)));

        CHECK(hipMemcpyAsync(d_qx, sys.qx.data(), sys.n * sizeof(double), hipMemcpyHostToDevice, stream));
        CHECK(hipMemcpyAsync(d_qy, sys.qy.data(), sys.n * sizeof(double), hipMemcpyHostToDevice, stream));
        CHECK(hipMemcpyAsync(d_qz, sys.qz.data(), sys.n * sizeof(double), hipMemcpyHostToDevice, stream));
        CHECK(hipMemcpyAsync(d_vx, sys.vx.data(), sys.n * sizeof(double), hipMemcpyHostToDevice, stream));
        CHECK(hipMemcpyAsync(d_vy, sys.vy.data(), sys.n * sizeof(double), hipMemcpyHostToDevice, stream));
        CHECK(hipMemcpyAsync(d_vz, sys.vz.data(), sys.n * sizeof(double), hipMemcpyHostToDevice, stream));
        CHECK(hipMemcpyAsync(d_m, sys.m.data(), sys.n * sizeof(double), hipMemcpyHostToDevice, stream));
        CHECK(hipMemcpyAsync(d_type, sys.type.data(), sys.n * sizeof(int), hipMemcpyHostToDevice, stream));
        
        int init_hit_step = -1;
        CHECK(hipMemcpyAsync(d_hit_step, &init_hit_step, sizeof(int), hipMemcpyHostToDevice, stream));
        
        int blockSize = 256;
        int numBlocks = (sys.n + blockSize - 1) / blockSize;

        for (int step = 0; step <= param::n_steps; step++) {
            if (step > 0) {
                compute_forces_update_v<<<numBlocks, blockSize, 0, stream>>>(sys.n, d_qx, d_qy, d_qz, d_vx, d_vy, d_vz, d_m, d_type, step * param::dt, sys.planet, sys.asteroid, nullptr);
                update_positions<<<numBlocks, blockSize, 0, stream>>>(sys.n, d_qx, d_qy, d_qz, d_vx, d_vy, d_vz);
            }
            check_collision_kernel<<<1, 1, 0, stream>>>(sys.planet, sys.asteroid, d_qx, d_qy, d_qz, d_hit_step, step);
        }
        
        int h_hit_step;
        CHECK(hipMemcpyAsync(&h_hit_step, d_hit_step, sizeof(int), hipMemcpyDeviceToHost, stream));
        CHECK(hipStreamSynchronize(stream));
        
        if (h_hit_step != -1) {
            hit_time_step = h_hit_step;
        }
        
        CHECK(hipFree(d_qx)); CHECK(hipFree(d_qy)); CHECK(hipFree(d_qz));
        CHECK(hipFree(d_vx)); CHECK(hipFree(d_vy)); CHECK(hipFree(d_vz));
        CHECK(hipFree(d_m)); CHECK(hipFree(d_type)); CHECK(hipFree(d_hit_step)); CHECK(hipFree(d_dummy_dist));
        CHECK(hipStreamDestroy(stream));
    });

    t1.join();
    t2.join();
    
    auto p2_end = std::chrono::high_resolution_clock::now();
    std::cerr << "Problem 1 & 2 time: " << std::chrono::duration<double>(p2_end - p1_start).count() << "s\n";

    // Problem 3
    int gravity_device_id = -1;
    double missile_cost = 0.0;

    if (hit_time_step > -1) {
        // Identify devices
        std::vector<int> devices;
        for(int i=0; i<sys.n; ++i) {
            if(sys.type[i] == 2) devices.push_back(i);
        }

        double best_cost = std::numeric_limits<double>::infinity();
        int best_id = -1;
        std::mutex result_mutex;

        auto worker = [&](int gpu_id, const std::vector<int>& device_subset) {
            CHECK(hipSetDevice(gpu_id));
            setup_gpu_constants();

            int blockSize = 256;
            int numBlocks = (sys.n + blockSize - 1) / blockSize;

            const int n_streams = 4;
            struct StreamCtx {
                hipStream_t stream;
                double *qx, *qy, *qz, *vx, *vy, *vz, *m, *dummy_dist;
                int *type, *hit_step, *destroyed_step;
                int *h_hit_step, *h_destroyed_step; // Pinned host memory
            };
            std::vector<StreamCtx> ctxs(n_streams);

            for(int i=0; i<n_streams; ++i) {
                CHECK(hipStreamCreate(&ctxs[i].stream));
                CHECK(hipMalloc(&ctxs[i].qx, sys.n * sizeof(double)));
                CHECK(hipMalloc(&ctxs[i].qy, sys.n * sizeof(double)));
                CHECK(hipMalloc(&ctxs[i].qz, sys.n * sizeof(double)));
                CHECK(hipMalloc(&ctxs[i].vx, sys.n * sizeof(double)));
                CHECK(hipMalloc(&ctxs[i].vy, sys.n * sizeof(double)));
                CHECK(hipMalloc(&ctxs[i].vz, sys.n * sizeof(double)));
                CHECK(hipMalloc(&ctxs[i].m, sys.n * sizeof(double)));
                CHECK(hipMalloc(&ctxs[i].type, sys.n * sizeof(int)));
                CHECK(hipMalloc(&ctxs[i].hit_step, sizeof(int)));
                CHECK(hipMalloc(&ctxs[i].destroyed_step, sizeof(int)));
                CHECK(hipMalloc(&ctxs[i].dummy_dist, sizeof(double)));
                CHECK(hipHostMalloc(&ctxs[i].h_hit_step, sizeof(int)));
                CHECK(hipHostMalloc(&ctxs[i].h_destroyed_step, sizeof(int)));
            }

            for (size_t i = 0; i < device_subset.size(); i += n_streams) {
                // Launch batch
                for (int s = 0; s < n_streams && (i + s) < device_subset.size(); ++s) {
                    int d_idx = device_subset[i+s];
                    StreamCtx& ctx = ctxs[s];

                    // Async copies
                    CHECK(hipMemcpyAsync(ctx.qx, sys.qx.data(), sys.n * sizeof(double), hipMemcpyHostToDevice, ctx.stream));
                    CHECK(hipMemcpyAsync(ctx.qy, sys.qy.data(), sys.n * sizeof(double), hipMemcpyHostToDevice, ctx.stream));
                    CHECK(hipMemcpyAsync(ctx.qz, sys.qz.data(), sys.n * sizeof(double), hipMemcpyHostToDevice, ctx.stream));
                    CHECK(hipMemcpyAsync(ctx.vx, sys.vx.data(), sys.n * sizeof(double), hipMemcpyHostToDevice, ctx.stream));
                    CHECK(hipMemcpyAsync(ctx.vy, sys.vy.data(), sys.n * sizeof(double), hipMemcpyHostToDevice, ctx.stream));
                    CHECK(hipMemcpyAsync(ctx.vz, sys.vz.data(), sys.n * sizeof(double), hipMemcpyHostToDevice, ctx.stream));
                    CHECK(hipMemcpyAsync(ctx.m, sys.m.data(), sys.n * sizeof(double), hipMemcpyHostToDevice, ctx.stream));
                    CHECK(hipMemcpyAsync(ctx.type, sys.type.data(), sys.n * sizeof(int), hipMemcpyHostToDevice, ctx.stream));
                    
                    CHECK(hipMemsetAsync(ctx.hit_step, 0xFF, sizeof(int), ctx.stream));
                    CHECK(hipMemsetAsync(ctx.destroyed_step, 0xFF, sizeof(int), ctx.stream));

                    for (int step = 0; step <= param::n_steps; step++) {
                        if (step > 0) {
                            compute_forces_update_v<<<numBlocks, blockSize, 0, ctx.stream>>>(sys.n, ctx.qx, ctx.qy, ctx.qz, ctx.vx, ctx.vy, ctx.vz, ctx.m, ctx.type, step * param::dt, sys.planet, sys.asteroid, nullptr);
                            update_positions<<<numBlocks, blockSize, 0, ctx.stream>>>(sys.n, ctx.qx, ctx.qy, ctx.qz, ctx.vx, ctx.vy, ctx.vz);
                        }
                        check_missile_kernel<<<1, 1, 0, ctx.stream>>>(sys.planet, d_idx, ctx.qx, ctx.qy, ctx.qz, ctx.m, ctx.type, step, ctx.destroyed_step);
                        check_collision_kernel<<<1, 1, 0, ctx.stream>>>(sys.planet, sys.asteroid, ctx.qx, ctx.qy, ctx.qz, ctx.hit_step, step);
                    }
                    
                    CHECK(hipMemcpyAsync(ctx.h_hit_step, ctx.hit_step, sizeof(int), hipMemcpyDeviceToHost, ctx.stream));
                    CHECK(hipMemcpyAsync(ctx.h_destroyed_step, ctx.destroyed_step, sizeof(int), hipMemcpyDeviceToHost, ctx.stream));
                }

                // Sync and process
                for (int s = 0; s < n_streams && (i + s) < device_subset.size(); ++s) {
                    StreamCtx& ctx = ctxs[s];
                    CHECK(hipStreamSynchronize(ctx.stream));
                    
                    int h_hit = *ctx.h_hit_step;
                    int h_destroyed = *ctx.h_destroyed_step;
                    
                    if (h_hit == -1) {
                        double current_cost = 0.0;
                        if (h_destroyed != -1) {
                            current_cost = param::get_missile_cost(h_destroyed * param::dt);
                        }
                        
                        std::lock_guard<std::mutex> lock(result_mutex);
                        if (current_cost < best_cost) {
                            best_cost = current_cost;
                            best_id = device_subset[i+s];
                        }
                    }
                }
            }
            
            // Cleanup
            for(int i=0; i<n_streams; ++i) {
                CHECK(hipFree(ctxs[i].qx)); CHECK(hipFree(ctxs[i].qy)); CHECK(hipFree(ctxs[i].qz));
                CHECK(hipFree(ctxs[i].vx)); CHECK(hipFree(ctxs[i].vy)); CHECK(hipFree(ctxs[i].vz));
                CHECK(hipFree(ctxs[i].m)); CHECK(hipFree(ctxs[i].type));
                CHECK(hipFree(ctxs[i].hit_step)); CHECK(hipFree(ctxs[i].destroyed_step));
                CHECK(hipFree(ctxs[i].dummy_dist));
                CHECK(hipHostFree(ctxs[i].h_hit_step));
                CHECK(hipHostFree(ctxs[i].h_destroyed_step));
                CHECK(hipStreamDestroy(ctxs[i].stream));
            }
        };

        std::vector<int> devices0, devices1;
        for (size_t i = 0; i < devices.size(); ++i) {
            if (i % 2 == 0) devices0.push_back(devices[i]);
            else devices1.push_back(devices[i]);
        }

        std::thread t0(worker, 0, devices0);
        std::thread t1(worker, 1, devices1);
        t0.join();
        t1.join();

        if (best_id != -1) {
            gravity_device_id = best_id;
            missile_cost = best_cost;
        }
    }

    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);
    
    auto p3_end = std::chrono::high_resolution_clock::now();
    std::cerr << "Problem 3 time: " << std::chrono::duration<double>(p3_end - p2_end).count() << "s\n";
    std::cerr << "Total time: " << std::chrono::duration<double>(p3_end - p1_start).count() << "s\n";

    return 0;
}
