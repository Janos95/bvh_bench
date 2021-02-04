#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"

#include "happly.h"

#include <bvh/bvh.hpp>
#include <bvh/vector.hpp>
#include <bvh/ray.hpp>
#include <bvh/sweep_sah_builder.hpp>
#include <bvh/locally_ordered_clustering_builder.hpp>
#include <bvh/linear_bvh_builder.hpp>
#include <bvh/single_ray_traverser.hpp>

#include "ScopedTimer.h"

using Scalar      = float;
using Vector3     = bvh::Vector3<Scalar>;
using BoundingBox = bvh::BoundingBox<Scalar>;
using Ray         = bvh::Ray<Scalar>;
using Bvh         = bvh::Bvh<Scalar>;

int main()
{
    happly::PLYData plyIn(ASSET_PATH);
    std::vector<std::array<double, 3>> pos;
    std::vector<std::vector<uint32_t>> indices;
    {
        ScopedTimer t{"Import", true};
        pos = plyIn.getVertexPositions();
        indices = plyIn.getFaceIndices<uint32_t>();
    }

    {
        ScopedTimer t{"Building Tree", true};

        size_t vCount = pos.size();
        size_t iCount = indices.size();
        size_t tCount = iCount/3;
        // The input of the BVH construction algorithm is just bounding boxes and centers
        std::vector<BoundingBox> bboxes(iCount);
        std::vector<Vector3> centers(iCount);

        BoundingBox global_bbox;

#pragma omp parallel for
        for(size_t i = 0; i < tCount; ++i) {
            uint32_t a = indices[i][0],b = indices[i][1],c = indices[i][2];
            BoundingBox box(Vector3(pos[a][0], pos[a][1], pos[a][2]));
            box.extend(Vector3(pos[b][0], pos[b][1], pos[b][2]));
            box.extend(Vector3(pos[c][0], pos[c][1], pos[c][2]));
            centers[i] = box.center();
            bboxes[i] = box;
        }

        Bvh bvh;

        bvh::LocallyOrderedClusteringBuilder<Bvh, size_t> builder(bvh);
        builder.build(global_bbox, bboxes.data(), centers.data(), bboxes.size());
    }
}
#pragma clang diagnostic pop
