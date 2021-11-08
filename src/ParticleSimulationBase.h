#pragma once

#include <array>
#include <eigen3/Eigen/Dense>
#include <random>
#include <iostream>
#include <chrono>
#include <vector>
#include <memory>
#include <algorithm>

using namespace Eigen;


class ParticleSimulationBase
{
    virtual void spawn() = 0;
    virtual void step(const std::size_t count = 1) = 0;
};
