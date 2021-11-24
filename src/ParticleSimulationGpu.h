#pragma once

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_CL_1_2_DEFAULT_BUILD



#include <fstream>
#include <CL/opencl.hpp>


#include "ParticleSimulationBase.h"


class ParticleSimulationGpu: public ParticleSimulationBase
{

public:
    static constexpr float G = 6.67E-7;
    static constexpr float e0 = 8.97E3;
    static constexpr float k0 = 1.0 / (4.0 * M_PI * e0 );
    static constexpr float fric = 0.999;

    static constexpr std::size_t particles = 1000;
    static constexpr float space_size = 3;
    static constexpr float dt = 0.001;
    std::array<cl_float4, particles> positions;
    std::array<cl_float4, particles> velocities;
    std::array<cl_float4, particles> accelerations;
    std::array<float, particles> masses;
    std::array<float, particles> sizes;


    cl::Buffer d_positions;
    cl::Buffer d_velocities;
    cl::Buffer d_accelerations;
    cl::Buffer d_masses;
    cl::Buffer d_sizes;
    // get default device and setup context
    cl::Platform m_platform;
    cl::Program m_program;
    cl::Context m_context;
    cl::CommandQueue m_queue;

    typedef cl::KernelFunctor<cl::Buffer,cl::Buffer, cl::Buffer,cl::Buffer,float> forces_kernel_t  ;
    std::unique_ptr<forces_kernel_t> m_forces_func;

    typedef cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer,float> integrate_kernel_t ;
    std::unique_ptr<integrate_kernel_t> m_integrate_func;



    typedef cl::KernelFunctor<cl::Buffer,cl::Buffer, float> limit_kernel_t;
    std::unique_ptr<limit_kernel_t> m_limit_func;

    typedef cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer,cl::Buffer, cl::Buffer> collisions_kernel_t;
    std::unique_ptr<collisions_kernel_t> m_collisions_func;

    std::string read_kernel(const std::string& path)
    {
        std::stringstream buffer;
        std::ifstream input_file(path);
        buffer << input_file.rdbuf();

        return buffer.str();

    }

    ParticleSimulationGpu(bool cpu = false)
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        for(auto &p: platforms)
        {
            std::string platver = p.getInfo<CL_PLATFORM_VERSION>();
            std::cout << p.getInfo<CL_PLATFORM_NAME>() << std::endl;
            std::cout << platver << std::endl;
        }
        m_platform = platforms[0];
        m_context = cl::Context(CL_DEVICE_TYPE_DEFAULT);
        m_queue = cl::CommandQueue(m_context);

        std::vector<std::string> kernels;
        kernels.push_back(read_kernel("kernels/kernel.cl"));

        m_program = cl::Program(m_context,kernels);
        try {
            std::cout << "Building program" << std::endl;
            m_program.build("-cl-std=CL1.2");
        }
        catch (...) {
            // Print build info for all devices
            cl_int buildErr = CL_SUCCESS;
            auto buildInfo = m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
            for (auto &pair : buildInfo) {
                std::cerr << pair.second << std::endl << std::endl;
            }
            exit(-1);
        }

        m_forces_func = std::make_unique<forces_kernel_t>(m_program, "forces");
        m_integrate_func = std::make_unique<integrate_kernel_t>(m_program, "integrate");
        m_limit_func = std::make_unique<limit_kernel_t>(m_program, "limit");
        m_collisions_func = std::make_unique<collisions_kernel_t>(m_program, "collisions");
    }

    void spawn()
    {
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(-2.0,2.0);

        for(std::size_t i = 0; i < particles; i++)
        {
            positions[i] = {distribution(generator), distribution(generator), distribution(generator)};
            velocities[i] =  {0,0.0,-0.1,0};
            accelerations[i] = {0,0,0,0};
            masses[i] = 10;
            sizes[i] = 0.05;
        }

        // masses[0] = 10000;

        d_positions = cl::Buffer(m_context, begin(positions), end(positions), false);
        d_velocities = cl::Buffer(m_context, begin(velocities), end(velocities),false);
        d_accelerations = cl::Buffer(m_context, begin(accelerations), end(accelerations),false);
        d_masses = cl::Buffer(m_context, begin(masses), end(masses),true);
        d_sizes = cl::Buffer(m_context, begin(sizes), end(sizes),true);


    }


    void step(const std::size_t count = 1)
    {
        const auto start_time = std::chrono::high_resolution_clock::now();
        try {
            for(std::size_t i = 0; i < count;i++)
            {

             //   (*m_forces_func)(cl::EnqueueArgs(m_queue,cl::NDRange(particles,particles)), d_positions, d_velocities, d_accelerations, d_masses, G);
                (*m_collisions_func)(cl::EnqueueArgs(m_queue,cl::NDRange(particles,particles)), d_positions, d_velocities, d_accelerations, d_sizes,d_masses);
                (*m_limit_func)(cl::EnqueueArgs(m_queue,particles), d_positions, d_velocities,space_size);
                (*m_integrate_func)(cl::EnqueueArgs(m_queue,particles), d_positions, d_velocities, d_accelerations, dt);
                m_queue.finish();
            }
            cl::copy(m_queue,d_positions,begin(positions), end(positions));

        }  catch (cl::Error & er) {
            std::cout << er.err() << ": " << er.what() << std::endl;
        }

        //
        const auto duration = std::chrono::high_resolution_clock::now() - start_time;
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(duration).count() << std::endl;

    }
};
