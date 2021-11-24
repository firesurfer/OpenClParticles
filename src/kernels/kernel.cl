
float4 gravity(const float4 position_1, const float4 position_2,  const float m1, const float m2, const float G)
{

    const float dis = distance(position_1, position_2);
    const float dis_2 = dis*dis;

    const float F = (G * m1 * m2)/dis_2;
    return F * (position_2- position_1)/dis;
}

bool collision_detect(const float4 position_1, const float4 position_2, const float size_1, const float size_2)
{
    const float dis = distance(position_1, position_2);
    return dis < size_1 + size_2;
}


__kernel void integrate(__global float4* position, __global float4* velocity, __global float4* acceleration, const float dt)
{
    const uint gid = get_global_id(0);

    velocity[gid].xyz += dt * acceleration[gid].xyz;
    position[gid].xyz += dt * velocity[gid].xyz;
}



__kernel void forces(__global const float4* position,
                     __global const float4* velocity,
                     __global float4* acceleration,
                     __global const float* masses,
                     const float G)
{
    const uint id1 = get_global_id(0);
    const uint id2 = get_global_id(1);



    if(id1 != id2)
    {
        const float4 Fg = gravity(position[id1], position[id2], masses[id1], masses[id2], G);
        acceleration[id1] = Fg / masses[id1];
    }

}

__kernel void collisions(__global const float4* position,
                         __global const float4* velocity,
                         __global float4* acceleration,
                         __global const float* sizes,
                         __global const float* masses)
{
    const uint id1 = get_global_id(0);
    const uint id2 = get_global_id(1);


    if(id1 != id2)
    {
        const bool collision = collision_detect(position[id1], position[id2], sizes[id1],sizes[id2]);
        if(collision)
        {
            const float m1 = masses[id1];
            const float m2 = masses[id2];
            const float m1_m2 = m1 +m2;
            const float dis = distance(position[id1], position[id2]);
            const float distance_factor = (sizes[id1] + sizes[id2])/dis;

            acceleration[id1] -=  normalize(velocity[id1]) *distance_factor;
            acceleration[id2] -= normalize(velocity[id2]) * distance_factor;

        }
    }


}

__kernel void limit(__global float4* position, __global float4* velocity, float size)
{
    uint gid = get_global_id(0);

    if(fabs(position[gid].x) > size)
    {
        velocity[gid].x *= -1;
    }
    if(fabs(position[gid].y) > size)
        velocity[gid].y *= -1;
    if(fabs(position[gid].z) > size)
        velocity[gid].z *= -1;
}

