#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace param {
const int n_steps = 200000;
const double dt = 60;
const double eps = 1e-3;
const double G = 6.674e-11;
double gravity_device_mass(double m0, double t) {
    return m0 + 0.5 * m0 * fabs(sin(t / 6000));
}
const double planet_radius = 1e7;
const double missile_speed = 1e6;
double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
}  // namespace param

void read_input(const char* filename, int& n, int& planet, int& asteroid,
    std::vector<double>& qx, std::vector<double>& qy, std::vector<double>& qz,
    std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz,
    std::vector<double>& m, std::vector<std::string>& type) {
    std::ifstream fin(filename);
    fin >> n >> planet >> asteroid;
    qx.resize(n);
    qy.resize(n);
    qz.resize(n);
    vx.resize(n);
    vy.resize(n);
    vz.resize(n);
    m.resize(n);
    type.resize(n);
    for (int i = 0; i < n; i++) {
        fin >> qx[i] >> qy[i] >> qz[i] >> vx[i] >> vy[i] >> vz[i] >> m[i] >> type[i];
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

void run_step(int step, int n, std::vector<double>& qx, std::vector<double>& qy,
    std::vector<double>& qz, std::vector<double>& vx, std::vector<double>& vy,
    std::vector<double>& vz, const std::vector<double>& m,
    const std::vector<std::string>& type) {
    // compute accelerations
    std::vector<double> ax(n), ay(n), az(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            double mj = m[j];
            if (type[j] == "device") {
                mj = param::gravity_device_mass(mj, step * param::dt);
            }
            double dx = qx[j] - qx[i];
            double dy = qy[j] - qy[i];
            double dz = qz[j] - qz[i];
            double dist3 =
                pow(dx * dx + dy * dy + dz * dz + param::eps * param::eps, 1.5);
            ax[i] += param::G * mj * dx / dist3;
            ay[i] += param::G * mj * dy / dist3;
            az[i] += param::G * mj * dz / dist3;
        }
    }

    // update velocities
    for (int i = 0; i < n; i++) {
        vx[i] += ax[i] * param::dt;
        vy[i] += ay[i] * param::dt;
        vz[i] += az[i] * param::dt;
    }

    // update positions
    for (int i = 0; i < n; i++) {
        qx[i] += vx[i] * param::dt;
        qy[i] += vy[i] * param::dt;
        qz[i] += vz[i] * param::dt;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }
    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<std::string> type;

    auto distance = [&](int i, int j) -> double {
        double dx = qx[i] - qx[j];
        double dy = qy[i] - qy[j];
        double dz = qz[i] - qz[j];
        return sqrt(dx * dx + dy * dy + dz * dz);
    };

    // Problem 1
    double min_dist = std::numeric_limits<double>::infinity();
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
    for (int i = 0; i < n; i++) {
        if (type[i] == "device") {
            m[i] = 0;
        }
    }
    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) {
            run_step(step, n, qx, qy, qz, vx, vy, vz, m, type);
        }
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        min_dist = std::min(min_dist, sqrt(dx * dx + dy * dy + dz * dz));
    }

    // Problem 2
    int hit_time_step = -2;
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) {
            run_step(step, n, qx, qy, qz, vx, vy, vz, m, type);
        }
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
            hit_time_step = step;
            break;
        }
    }

    // Problem 3
    // TODO
    int gravity_device_id = -999;
    double missile_cost = -999;
    // If there is no collision in Problem 2, no missile is needed.
    if (hit_time_step <= -1) {
        gravity_device_id = -1;
        missile_cost = 0.0;
    } else {
        // Try destroying each device (one at a time) and simulate whether
        // the planet is saved. Choose the device with lowest missile cost
        // that prevents collision.
        double best_cost = std::numeric_limits<double>::infinity();
        int best_id = -1;

        // read initial state again so we have untouched starting values
        read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);

        for (int d = 0; d < n; d++) {
            if (type[d] != "device") continue;

            // copy initial state for simulation
            std::vector<double> qx2 = qx, qy2 = qy, qz2 = qz;
            std::vector<double> vx2 = vx, vy2 = vy, vz2 = vz;
            std::vector<double> m2 = m;
            std::vector<std::string> type2 = type;

            bool destroyed = false;
            bool collided = false;
            double cost_for_this = 0.0;

            for (int step = 0; step <= param::n_steps; step++) {
                if (step > 0) {
                    run_step(step, n, qx2, qy2, qz2, vx2, vy2, vz2, m2, type2);
                }

                // missile launched at start; check hit after positions updated
                if (!destroyed) {
                    // compute distance between planet and device at current step
                    double dx = qx2[planet] - qx2[d];
                    double dy = qy2[planet] - qy2[d];
                    double dz = qz2[planet] - qz2[d];
                    double dist_planet_device = sqrt(dx * dx + dy * dy + dz * dz);

                    double missile_traveled = step * param::dt * param::missile_speed;
                    if (missile_traveled > dist_planet_device) {
                        // device is destroyed at this step
                        destroyed = true;
                        type2[d] = "destroyed";
                        m2[d] = 0.0;
                        cost_for_this = param::get_missile_cost(step * param::dt);
                    }
                }

                // after possible destruction, check planet-asteroid collision
                double dxp = qx2[planet] - qx2[asteroid];
                double dyp = qy2[planet] - qy2[asteroid];
                double dzp = qz2[planet] - qz2[asteroid];
                if (dxp * dxp + dyp * dyp + dzp * dzp < param::planet_radius * param::planet_radius) {
                    collided = true;
                    break;
                }
            }

            if (!collided) {
                if (cost_for_this < best_cost) {
                    best_cost = cost_for_this;
                    best_id = d;
                }
            }
        }

        if (best_id == -1) {
            gravity_device_id = -1;
            missile_cost = 0.0;
        } else {
            gravity_device_id = best_id;
            missile_cost = best_cost;
        }
    }

    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);
}
