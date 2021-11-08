
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



__kernel void forces(__global const float4* position, __global const float4* velocity, __global float4* acceleration, __global const float* masses,const float G, uint num)
{
    uint gid = get_global_id(0);

    for(uint j = 0; j < num;j++)
    {
        if(j != gid)
        {
            const float4 Fg = gravity(position[gid], position[j], masses[gid], masses[j], G);
            acceleration[gid] = Fg / masses[gid];
        }
    }
}

__kernel void collisions(__global const float4* position, __global const float4* velocity, __global float4* acceleration, __global const float* sizes, __global const float* masses, const uint num)
{
    const uint gid = get_global_id(0);

    for(uint j = 0; j < num;j++)
    {
        if(j != gid)
        {
            const bool collision = collision_detect(position[gid], position[j], sizes[gid],sizes[j]);
            if(collision)
            {
                const float m1 = masses[gid];
                const float m2 = masses[j];
                const float m1_m2 = m1 +m2;
                const float dis = distance(position[j], position[gid]);
                const float distance_factor = (sizes[gid] + sizes[j])/dis;

                acceleration[gid] -=  normalize(velocity[gid]) *distance_factor;
                acceleration[j] -= normalize(velocity[j]) * distance_factor;

            }
        }
    }

}

__kernel void limit(__global const float4* position, __global float4* velocity, float size)
{
    uint gid = get_global_id(0);

    if(fabs(position[gid].x) > size)
        velocity[gid].x *= -1;
    if(fabs(position[gid].y) > size)
        velocity[gid].y *= -1;
    if(fabs(position[gid].z) > size)
        velocity[gid].z *= -1;
}

__kernel void steps(__global float4* position, __global float4* velocity, __global float4* acceleration, __global const float* masses, __global const float* sizes,const float G,const float dt, const uint step_count, uint num)
{
    for(uint i = 0; i < step_count;i++)
    {
        forces(position, velocity, acceleration,masses, G, num);
        //barrier(CLK_GLOBAL_MEM_FENCE);
        collisions(position, velocity, acceleration, sizes,masses, num);
       // barrier(CLK_GLOBAL_MEM_FENCE);
        limit(position, velocity, 3);
      //  barrier(CLK_GLOBAL_MEM_FENCE);
        integrate(position, velocity, acceleration,dt);
       // barrier(CLK_GLOBAL_MEM_FENCE);
    }
}
