#include "mesh_export.h"

#include <fstream>

#include <libutils/rasserts.h>


void phg::exportMesh(const std::string &path,
                     const std::vector<cv::Vec3d> &mesh_vertices, const std::vector<cv::Vec3i> &mesh_faces, const std::vector<cv::Vec3b> &mesh_vertices_color)
{
    size_t nvertices = mesh_vertices.size();
    size_t nfaces = mesh_faces.size();

    bool with_color = mesh_vertices_color.size() > 0;
    if (with_color) {
        rassert(mesh_vertices_color.size() == nvertices, 237821312011);
    }

    std::ofstream filestream(path);

    filestream << "ply" << std::endl;
    filestream << "format binary_little_endian 1.0" << std::endl;
    filestream << "element vertex " << nvertices << std::endl;
    filestream << "property float x" << std::endl;
    filestream << "property float y" << std::endl;
    filestream << "property float z" << std::endl;
    if (with_color) {
        filestream << "property uchar red" << std::endl;
        filestream << "property uchar green" << std::endl;
        filestream << "property uchar blue" << std::endl;
    }
    filestream << "element face " << nfaces << std::endl;
    filestream << "property list uchar int vertex_indices" << std::endl;
    filestream << "end_header" << std::endl;

    for (size_t vi = 0; vi < nvertices; ++vi) {
        cv::Vec3f p = mesh_vertices[vi];
        for (int i = 0; i < 3; ++i) {
            rassert(p[i] == p[i], 2389213134039);
        }
        filestream.write((const char *) (&(p[0])), 3 * sizeof(float));
        if (with_color) {
            cv::Vec3b color = mesh_vertices_color[vi];
            filestream.write((const char *) (&(color[0])), 3 * sizeof(unsigned char));
        }
    }
    for (size_t fi = 0; fi < nfaces; ++fi) {
        cv::Vec3i face = mesh_faces[fi];

        const unsigned char nvertices_per_face = 3;
        filestream.write((const char *) (&nvertices_per_face), 1 * sizeof(unsigned char));

        for (int i = 0; i < nvertices_per_face; ++i) {
            rassert(face[i] >= 0 && face[i] < nvertices, 23892131052);
        }
        filestream.write((const char *) (&(face[0])), nvertices_per_face * sizeof(int));
    }

    filestream.close();
}
