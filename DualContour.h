#ifndef DUALCONTOUR_H

#define DUALCONTOUR_H
#include <Eigen/Core>
#include <vector>

typedef struct {
  float x;
  float y;
  float z;
  float xn;
  float yn;
  float zn;
} vert;

typedef struct {
  vert verts[3];
} tri;

typedef struct {
  Eigen::Vector3f center;
  float dx;
  float dy;
  float dz;
} box;

typedef struct {
  Eigen::Vector3f center;
  float radius;
} sphere;

extern box UNIT_BOX;
extern box UNIT_BOUNDING_BOX;
extern sphere UNIT_SPHERE;

extern box DEFAULT_BOX;

float sphereDensity(sphere *s, Eigen::Vector3f p);
Eigen::Vector3f sphereNormal(sphere *s, Eigen::Vector3f p);
float boxDensity(box *b, Eigen::Vector3f p);
Eigen::Vector3f boxNormal(box *b, Eigen::Vector3f p);

typedef float (*densityFunction)(void *, Eigen::Vector3f);
typedef Eigen::Vector3f (*normalFunction)(void *, Eigen::Vector3f);

/*
 * Generate a vector of tri's given a density and normal function defined over
 * some bounds. density_arg is passed to the density function, and can be null.
 * normal_arg is passed to the normal function, and can be null.
 */
std::vector<tri> *dual_contour(box bounds, int resolution, void *density_arg,
                               densityFunction density, void *normal_arg,
                               normalFunction normal);

#endif /* end of include guard: DUALCONTOUR_H */
