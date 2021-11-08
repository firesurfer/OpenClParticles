#include <Open3D/Open3D.h>
#include "ParticleSimulationCpu.h"





int main(int argc, char** argv)
{

    open3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow();


    ParticleSimulationCpu sim;
    sim.spawn();
    std::vector<std::shared_ptr<open3d::geometry::TriangleMesh>> meshes;
    for(size_t i = 0; i < sim.particles;i++)
    {
        meshes.push_back(open3d::geometry::TriangleMesh::CreateSphere(sim.sizes[i]));
        meshes[i]->Translate(sim.positions[i].cast<double>());
        meshes[i]->ComputeVertexNormals();
        meshes[i]->PaintUniformColor({0.9,std::abs(sim.positions[i].z())/(2*sim.space_size),0});
        vis.AddGeometry(meshes[i]);
    }
        meshes[0]->PaintUniformColor({0.0,0.5,0});
    auto coord_frame = open3d::geometry::TriangleMesh::CreateCoordinateFrame();
    vis.AddGeometry(coord_frame);

    auto bound_box = open3d::geometry::TriangleMesh::CreateBox(2*sim.space_size,2* sim.space_size,2*sim.space_size);
    bound_box->Translate({-sim.space_size, -sim.space_size, -sim.space_size});
    auto bound_box_wireframe = open3d::geometry::LineSet::CreateFromTriangleMesh(*bound_box);
    vis.AddGeometry(bound_box_wireframe);

    auto start_time = std::chrono::high_resolution_clock::now();
    for(std::size_t i = 0; i < 500000000;i++)
    {

        sim.step();
        if(i % (std::size_t)(1.0/sim.dt) == 0)
        {
            std::cout  << "Step: " << i << std::endl;
            for(size_t i = 0; i < sim.particles;i++)
            {
              // std::cout << sim.cloud.vels[i] << std::endl;


                meshes[i]->Translate( sim.positions[i].cast<double>() -meshes[i]->GetCenter() );
               // meshes[i]->ComputeVertexNormals();
                //meshes[i]->PaintUniformColor({0.9,1,0});

                vis.UpdateGeometry(meshes[i]);

               }



            vis.PollEvents();
            vis.UpdateRender();

            auto duration =  std::chrono::high_resolution_clock::now() - start_time;
            start_time =  std::chrono::high_resolution_clock::now();

            std::cout << "1.0s took: " << std::chrono::duration_cast<std::chrono::microseconds>(duration).count() << "us" << std::endl;

        }

    }

    while(true)
    {
        vis.PollEvents();
        vis.UpdateRender();
    }

}


