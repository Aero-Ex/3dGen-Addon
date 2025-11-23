// main.cpp
#include <igl/read_triangle_mesh.h>
#include <igl/write_triangle_mesh.h>
#include <igl/per_face_normals.h>
#include <igl/adjacency_list.h>
#include <igl/lscm.h>

// LSCM and Mesh-related includes
#include "Mesh.h"
#include "FormTrait.h"
#include "LSCM.h"

// Triangle intersection
#include "triangleHelper.hpp"

// Distortion computation

#include <Eigen/Dense>
#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <filesystem>  // C++17 feature

// Unwrapping and component segmentation
#include "UnwrapBB.h"
#include "UnwrapMerge.h"
#include "Component.h"
#include "Distortion.h"
#include "pipeline.h"

// Merge and IO routines
#include "merge.h"
#include "IO.h"

// Enable profiling and save options
#define ENABLE_PROFILING



#include "Config.h"
#include "Pack.h"

using namespace MeshLib;
namespace fs = std::filesystem;

int main(int argc, char *argv[])
{

    std::string binPath, meshPath, configPath;
    std::string meshOverride;  // will hold the override mesh path if provided

    // Determine the config file to load and the mesh override.
    // Usage: <program> <override_meshpath> <config> <override_threshold>
    configPath = "../../config/config.yaml";
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <override_meshpath> <config> <override_outputDir>" << std::endl;
        std::cout << "No mesh override provided, using mesh path from config after loading default config file." << std::endl;
    } else {
        // First argument is the mesh override
        meshOverride = argv[1];
        // Second argument (if provided) is the config file override
        if (argc >= 3 && !std::string(argv[2]).empty()) {
             configPath = argv[2];
        } 
    }
    
    // Now load the config file so that CONFIG_meshPath is set.
    ConfigManager::instance().loadFromFile(configPath);


    if(!meshOverride.empty()) ConfigManager::instance().setMeshPath(meshOverride);

    ConfigManager::instance().printConfigs();
    
    std::string exp = CONFIG_meshPath;
    double threshold = CONFIG_pipelineThreshold;
    

    size_t lastSlashPos = exp.find_last_of('/');
    std::string directory = exp.substr(0, lastSlashPos);
    std::string filename = exp.substr(lastSlashPos + 1);
    
    size_t dotPos = filename.find_last_of('.');
    std::string stem = filename.substr(0, dotPos);
    

    binPath = directory + "/bin/" + stem + ".bin";
    meshPath = exp;
    

    std::string output_directory = CONFIG_outputPath;
    if (output_directory.empty()) {
        output_directory = directory + "/output/";
    }

    // Create the output directory if it doesn't exist
    fs::path output_path = output_directory;

    CONFIG_outputPath = output_path.string();

    fs::create_directories(output_directory);



    std::vector<UVParts> individual_parts;  
    UVParts final_part = pipeline(binPath, meshPath, threshold, individual_parts);
    
    Eigen::MatrixXd Nc;  Eigen::MatrixXi FNc; // (empty) vertex normals and face normals for OBJ
    Eigen::MatrixXi FUVc; 
    
    int total_components = 0;

    // IglUvWrapper wrapper;

    for(int i = 0; i < individual_parts.size(); i++){
        UVParts part = individual_parts[i];
        total_components += part.num_components;
        // for (Component chart : part.components){
            // normalize_uv_by_3d_area(chart);
            // wrapper.add_component(chart, /*group_id=*/i, false);
        // }

        // wrapper.add_component(comp, i);
        Component comp = part.to_components();

        fs::path submesh_path = output_path / ("part_" + std::to_string(i) + ".obj");

        FUVc = comp.F;
        if (!igl::writeOBJ(submesh_path, comp.V, comp.F, Nc, FNc, comp.UV, FUVc)){
            std::cerr << "Failed to write submesh OBJ: " << submesh_path << std::endl;
        }
    }

    
    // wrapper.add_component(UV_component, 0, false);
    // // Suppress output from packing
    // std::cout.setstate(std::ios_base::failbit);
    // UV_component.UV = wrapper.runPacking();
    // std::cout.clear();
    // save_uv_layout(UV_component.UV, UV_component.F, "uv_layout_uvp_after.png");
    
    
    Component UV_component = final_part.to_components();

    bool has_overlaps = false;
    std::vector<std::pair<int,int>> all_overlappingTriangles;
    // for(int i = 0; i < std::min(50, (int)individual_parts.size()); i++){
    for(int i = 0; i < individual_parts.size(); i++){
        UVParts part = individual_parts[i];
        Component comp = part.to_components();
        std::cout << "Part " << i << " has " << part.num_components << " charts and distortion " << comp.distortion << std::endl;

        for (Component chart : part.components){
            std::vector<std::pair<int,int>> overlappingTriangles;
            if(computeOverlapingTrianglesFast(chart.UV, chart.F,overlappingTriangles)){
                std::cout << "Overlapping triangles detected in part " << i << std::endl;
                has_overlaps = true;
                all_overlappingTriangles.insert(all_overlappingTriangles.end(), overlappingTriangles.begin(), overlappingTriangles.end());
            }
        } 
    }



    std::cout << "# of V: " << UV_component.V.rows() << std::endl;
    std::cout << "# of F: " << UV_component.F.rows() << std::endl;
    std::cout << "# of UV: " << UV_component.UV.rows() << std::endl;
    std::cout << "Total components: " << total_components << std::endl;
    std::cout << "Total overlapping triangles: " << all_overlappingTriangles.size() << std::endl;

    Eigen::MatrixXd UVc = UV_component.UV;
    Eigen::MatrixXd Vc = UV_component.V;
    Eigen::MatrixXi Fc = UV_component.F;
    
    fs::path combined_mesh_path = output_path / "final_components.obj";

    if (!igl::writeOBJ(combined_mesh_path, Vc, Fc, Nc, FNc, UVc, Fc)) {
        std::cerr << "Failed to write combined OBJ: " << combined_mesh_path << std::endl;
    } else {
        std::cout << "Wrote combined OBJ: " << combined_mesh_path << std::endl;
    }



}
