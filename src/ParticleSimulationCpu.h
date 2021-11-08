#pragma once

#include "ParticleSimulationBase.h"


class ParticleSimulationCpu: public ParticleSimulationBase
{

public:
    static constexpr float G = 6.67E-5;
    static constexpr float e0 = 8.97E3;
    static constexpr float k0 = 1.0 / (4.0 * M_PI * e0 );
    static constexpr float fric = 0.999;

    static constexpr std::size_t particles = 200;
    static constexpr float space_size = 3;
    static constexpr float dt = 0.0001;
    std::array<Vector3f, particles> positions;
    std::array<Vector3f, particles> velocities;
    std::array<Vector3f, particles> accelerations;
    std::array<float, particles> masses;
    std::array<float, particles> sizes;

    void spawn()
    {
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(-2.0,2.0);

        for(std::size_t i = 0; i < particles; i++)
        {
            positions[i] = {distribution(generator), distribution(generator), distribution(generator)};
            velocities[i] = {0,0.0,0};
            accelerations[i] = {0,0,0};
            masses[i] = 10;
            sizes[i] = 0.05;
        }
    }

    Vector3f gravity(const std::size_t& ia,const std::size_t& ib) const
    {
        const auto m1 = masses[ia];
        const auto m2 = masses[ib];

        const auto diff = positions[ib] - positions[ia];
        const auto distance = diff.norm();
        const auto distance_2 = distance * distance;

        const auto F = (G* m1 * m2)/distance_2;
        return F * diff/distance;
    }

    void integrate(std::size_t& ia)
    {
        velocities[ia] += accelerations[ia] * dt;
        positions[ia] += velocities[ia] * dt;
    }

    void limit(std::size_t& ia)
    {
        const auto pos = positions[ia];

        if(std::abs(pos.x()) > space_size - sizes[ia])
            velocities[ia].x() *= -1;
        if(std::abs(pos.y()) > space_size - sizes[ia])
            velocities[ia].y() *= -1;
        if(std::abs(pos.z()) > space_size - sizes[ia])
            velocities[ia].z() *= -1;
    }

    std::tuple<float,bool> collision_detect(const std::size_t& ia,const std::size_t& ib) const
    {
        const auto distance =(positions[ib] - positions[ia]).norm();
        return {distance,distance < sizes[ia] + sizes[ib]};
    }

    void collision_change(const std::size_t& ia,const std::size_t& ib)
    {
        const auto m1 = masses[ia];
        const auto m2 = masses[ib];
        const auto m1_m2 = m1+m2;
        const auto diff = positions[ib] - positions[ia];
        const auto distance = diff.norm();
        const auto distance_factor = (sizes[ia]+ sizes[ib])/distance;

        const auto Ekin_1 = 0.5 * m1 * velocities[ia].cwiseProduct(velocities[ia]);
        const auto Ekin_2 = 0.5 * m2 * velocities[ib].cwiseProduct(velocities[ib]);

        const auto v_c = 2 * (m1 * velocities[ia] + m2 * velocities[ib]) / m1_m2;
        accelerations[ia] -= 10* m1 * velocities[ia].normalized()*distance_factor;
        accelerations[ib] -= 10* m2 * velocities[ib].normalized()*distance_factor;
        //velocities[ia] =fric*( v_c - velocities[ia]);
        //velocities[ib] = fric*(-1 * (v_c - velocities[ib]));
    }
    void interact(const std::size_t& ia,const std::size_t& ib)
    {
        const auto Fg = gravity(ia,ib);
        accelerations[ia] = Fg / masses[ia];
        accelerations[ib] = -Fg / masses[ib];
    }

    void step(const std::size_t count = 1)
    {
        const auto start_time = std::chrono::high_resolution_clock::now();
        //#pragma omp parallel for
        for(std::size_t i = 0; i < particles;i++)
        {
            for(std::size_t j =i+1; j < particles;j++)
            {

                if(i != j)
                {
                    interact(i,j);
                }
            }
        }


        //#pragma omp parallel for
        for(std::size_t i = 0; i < particles;i++)
        {
            for(std::size_t j =i+1; j < particles;j++)
            {

                if(i != j)
                {
                    const auto [distance, collision] = collision_detect(i,j);
                            if(collision)
                    {
                        //std::cout << distance << std::endl;
                        collision_change(i,j);
                    }
                }
            }
        }
        //#pragma omp parallel for
        for(std::size_t i = 0; i < particles;i++)
        {
            integrate(i);
            limit(i);
        }

        const auto duration = std::chrono::high_resolution_clock::now() - start_time;
        // std::cout << std::chrono::duration_cast<std::chrono::microseconds>(duration).count() << std::endl;

    }
};
