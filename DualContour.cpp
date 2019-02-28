// Copyright K.R. Zentner, 2013.
#include <DualContour.h>
#include <Eigen/SVD>
#include <cmath>
#include <iostream>

using namespace Eigen;

box UNIT_BOX = {
    Vector3f(0, 0, 0),
    0.5,
    0.5,
    0.5,
};

box UNIT_BOUNDING_BOX = {
    Vector3f(0, 0, 0),
    0.75,
    0.75,
    0.75,
};

box DEFAULT_BOX = {
    Vector3f(0, 0, 0),
    6.0,
    6.0,
    6.0,
};

sphere UNIT_SPHERE = {
    Vector3f(0, 0, 0),
    0.25,
};

sphere DEFAULT_SPHERE = {
    Vector3f(0, 0, 0),
    5.9,
};

float sphereDensity(sphere *s, Vector3f p) {
  Vector3f d = p - s->center;
  return d.norm() - s->radius;
}

Vector3f sphereNormal(sphere *s, Vector3f p) {
  return (p - s->center).normalized();
}

float boxDensity(box *b, Vector3f p) {
  if (b->center.x() - b->dx < p.x() && b->center.x() + b->dx > p.x() &&
      b->center.y() - b->dy < p.y() && b->center.y() + b->dy > p.y() &&
      b->center.z() - b->dz < p.z() && b->center.z() + b->dz > p.z()) {
    // Inside
    return -1.0;
  } else {
    // Outside
    return 1.0;
  }
}

static float sign(float in) {
  if (in > 0) {
    return 1.0;
  } else {
    return -1.0;
  }
}

Vector3f boxNormal(box *b, Vector3f p) {
  Vector3f diff = p - b->center;
  float x = fabs(diff.x() / b->dx);
  float y = fabs(diff.y() / b->dy);
  float z = fabs(diff.z() / b->dz);
  if (x > y && x > z) {
    return Vector3f(sign(diff.x()), 0, 0);
  } else if (y > x && y > z) {
    return Vector3f(0, sign(diff.y()), 0);
  } else {
    return Vector3f(0, 0, sign(diff.z()));
  }
}

#define INTERSECTION_ITERS 4
static Vector3f find_intersection(void *density_arg, densityFunction density,
                                  Vector3f inside, Vector3f outside) {
  if (density(density_arg, outside) < 0) {
    // The outside point is actually inside.
    // Swap the points.
    Vector3f temp = inside;
    inside = outside;
    outside = temp;
  }
  // Use binary search to minimize density function.
  float t = 0.0;
  float dt = 0.5;
  Vector3f d = outside - inside;
  for (int i = 0; i < INTERSECTION_ITERS; i++) {
    Vector3f next = d * (t + dt) + inside;
    if (density(density_arg, next) < 0) {
      t += dt;
      dt /= 2.0;
    } else {
      dt /= 2.0;
    }
  }
  return d * t + inside;
}

// Vertices like (0, 0, 1), (0, 1, 0), (0, 1, 1), etc.
static Vector3i unit_cube_verts[8];

// Pairs of indices into unit_cube_verts corresponding to the edges of the
// cube.
static int cube_edges[24][2];

static void initialize_dual_contour() {
  for (int x = 0; x < 2; x++) {
    for (int y = 0; y < 2; y++) {
      for (int z = 0; z < 2; z++) {
        unit_cube_verts[x * 4 + y * 2 + z] = Vector3i(x, y, z);
      }
    }
  }

  // Initialize cube_edges.
  int n = 0;
  int m = 0;
  for (int a = 0; a < 2; a++) {
    for (int b = 0; b < 2; b++) {
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          if (j == i) {
            // Cube edges don't start and end at the same vertex.
            continue;
          }
          for (int k = 0; k < 8; k++) {
            Vector3f v = unit_cube_verts[k].cast<float>();
            if (v[i] == a && v[j] == b) {
              cube_edges[n][m++] = k;
            }
          }
          ++n;
          m = 0;
        }
      }
    }
  }
}

static tri make_tri(Vector3f a, Vector3f an, Vector3f b, Vector3f bn,
                    Vector3f c, Vector3f cn) {
  tri out;

  out.verts[0].x = a.x();
  out.verts[0].y = a.y();
  out.verts[0].z = a.z();

  out.verts[1].x = b.x();
  out.verts[1].y = b.y();
  out.verts[1].z = b.z();

  out.verts[2].x = c.x();
  out.verts[2].y = c.y();
  out.verts[2].z = c.z();

  out.verts[0].xn = an.x();
  out.verts[0].yn = an.y();
  out.verts[0].zn = an.z();

  out.verts[1].xn = bn.x();
  out.verts[1].yn = bn.y();
  out.verts[1].zn = bn.z();

  out.verts[2].xn = cn.x();
  out.verts[2].yn = cn.y();
  out.verts[2].zn = cn.z();

  return out;
}

std::vector<tri> *dual_contour(box bounds, const int resolution,
                               void *density_arg, densityFunction density,
                               void *normal_arg, normalFunction normal) {
  std::vector<tri> *out = new std::vector<tri>();
  initialize_dual_contour();

  const float x_start = bounds.center.x() - bounds.dx;
  const float dx = 2 * bounds.dx / resolution;
  const float y_start = bounds.center.y() - bounds.dy;
  const float dy = 2 * bounds.dy / resolution;
  const float z_start = bounds.center.z() - bounds.dz;
  const float dz = 2 * bounds.dz / resolution;

  // These are the vertices of a cube the size of a single grid cell, with one
  // corner at the origin.
  Vector3f d_cube_verts[8];
  for (int x = 0; x < 2; x++) {
    for (int y = 0; y < 2; y++) {
      for (int z = 0; z < 2; z++) {
        d_cube_verts[x * 4 + y * 2 + z] = Vector3f(x * dx, y * dy, z * dz);
      }
    }
  }

  // These need to be C-style buffers, not std::vector's, since they are
  // accessed from multiple threads.
#define SOLUTION(x, y, z)                                                      \
  (solutions[(x) + resolution * ((y) + resolution * (z))])
#define HAS_SOLUTION(x, y, z) (SOLUTION((x), (y), (z)) != Vector3f(0, 0, 0))
  Vector3f *solutions = (Vector3f *)malloc(sizeof(Vector3f) * resolution *
                                           resolution * resolution);

  // Note that this grid is of size ( resolution + 1 ) because there are one
  // more corners per dimension than there are cells.
#define INSIDE(x, y, z)                                                        \
  (inside_corners[(x) + resolution * ((y) + resolution * (z))])
  bool *inside_corners = (bool *)malloc(sizeof(bool) * (resolution + 1) *
                                        (resolution + 1) * (resolution + 1));

  if (!solutions || !inside_corners) {
    std::cout << "DualContouring could not allocate buffer." << std::endl;
    free(solutions);
    free(inside_corners);
    return out;
  }

  // Calculate which corners are inside the surface.
#pragma omp parallel for
  for (int z = 0; z <= resolution; z++) {
    for (int y = 0; y <= resolution; y++) {
      for (int x = 0; x <= resolution; x++) {
        Vector3f origin = Vector3f(x_start + (dx * x), y_start + (dy * y),
                                   z_start + (dz * z));
        INSIDE(x, y, z) = density(density_arg, origin) > 0;
      }
    }
  }

#pragma omp parallel for
  for (int z = 0; z < resolution; z++) {
    for (int y = 0; y < resolution; y++) {
      for (int x = 0; x < resolution; x++) {
        Vector3f origin = Vector3f(x_start + (dx * x), y_start + (dy * y),
                                   z_start + (dz * z));
        int cube_signs = 0;
        for (int i = 0; i < 8; i++) {
          Vector3i corner = unit_cube_verts[i] + Vector3i(x, y, z);
          if (!INSIDE(corner.x(), corner.y(), corner.z())) {
            // Vertex i is outside the surface.
            cube_signs |= 0x1 << i;
          }
        }

        if (cube_signs == 0 || cube_signs == 0xff) {
          // The cube was all outside or all inside.
          SOLUTION(x, y, z) = Vector3f(0, 0, 0);
          continue;
        }

        int edge_count = 0;
        for (int i = 0; i < 24; i++) {
          int vert_0 = cube_edges[i][0];
          int vert_1 = cube_edges[i][1];
          if (!(cube_signs & (0x1 << vert_0)) !=
              !(cube_signs & (0x1 << vert_1))) {
            edge_count += 1;
          }
        }

        // Calculate the intersection for each edge.
        Matrix<float, Dynamic, 3> A(edge_count, 3);
        Matrix<float, Dynamic, 1> b(edge_count);

        Matrix<float, Dynamic, 3> intersections(edge_count, 3);

        Vector3f midpoint = Vector3f(0, 0, 0);

        int j = 0;
        for (int i = 0; i < 24; i++) {
          int vert_0 = cube_edges[i][0];
          int vert_1 = cube_edges[i][1];
          if (!(cube_signs & (0x1 << vert_0)) ==
              !(cube_signs & (0x1 << vert_1))) {
            // There was no sign change, skip this edge.
            SOLUTION(x, y, z) = Vector3f(0, 0, 0);
            continue;
          }

          Vector3f inside = origin + d_cube_verts[vert_0];
          Vector3f outside = origin + d_cube_verts[vert_1];

          Vector3f the_intersection =
              find_intersection(density_arg, density, inside, outside);

          Vector3f the_normal = normal(normal_arg, the_intersection);

          A.row(j) = the_normal;
          intersections.row(j) = the_intersection;

          ++j;
        }

        // Calculate the midpoint.
        for (int i = 0; i < j; i++) {
          midpoint += intersections.row(i);
        }

        midpoint /= j;

        // Center the solution domain about the midpoint.
        for (int i = 0; i < j; i++) {
          Vector3f the_intersection = intersections.row(i);
          b.row(i) << (the_intersection - midpoint).dot(A.row(i));
        }

        JacobiSVD<Matrix<float, Dynamic, 3>, HouseholderQRPreconditioner> svd(
            A, ComputeFullU | ComputeFullV);

        Matrix<float, Dynamic, Dynamic> U = svd.matrixU();
        Matrix<float, 3, 3> V = svd.matrixV();
        Matrix<float, 3, 1> S = svd.singularValues();

        const int num_singular_vals = 3;
        for (int s = 0; s < num_singular_vals; s++) {
          // Remove dimensions of the pseudo-inverse which would be too large.
          // This prevents unstable solutions.
          // Note that this causes those dimensions to be clipped to zero,
          // which is what we want since we centered the solution domain about
          // the midpoint above.
          if (S[s] < 0.1) {
            S[s] = 0;
          } else {
            S[s] = 1 / S[s];
          }
        }

        // E is the diagonal matrix of the pseudo-inverse.
        // However, it is not square (in this design of the algorithm).
        Matrix<float, Dynamic, 3> E(U.cols(), 3);
        E = E.Zero(U.cols(), 3);
        E.diagonal() = S;

        // Calculate the pseudo-inverse of A, and solve for b.
        Vector3f solution = V * E.transpose() * U.transpose() * b;

        // Use binary search to find the actual location of the surface near
        // the solution.
        // This is an extension of Dual Contouring which makes it more robust
        // when the density function tends to have details which are smaller
        // than the grid itself.
        // For example, this prevents single particles in a grid of a higher
        // resolution than the grid from producing spikes.

        SOLUTION(x, y, z) = find_intersection(density_arg, density, midpoint,
                                              midpoint + solution);

        // Move the solution (which is relative to the midpoint) back in place.
        // For standar Dual Contouring, comment the above line and uncomment
        // the next line.

        // SOLUTION ( x, y, z ) = solution + midpoint;
      }
    }
  }

#define NORMAL(x, y, z)                                                        \
  (normal(normal_arg, Vector3f((x_start + dx * (x)), (y_start + dy * (y)),     \
                               (z_start + dz * (z)))))

  // Output quads (pairs of tris) for each 'higher' neighbor who also has a
  // solution.
  for (int z = 0; z < resolution; z++) {
    for (int y = 0; y < resolution; y++) {
      for (int x = 0; x < resolution; x++) {
        if (!HAS_SOLUTION(x, y, z)) {
          continue;
        }
        if (HAS_SOLUTION(x, y + 1, z) && HAS_SOLUTION(x + 1, y, z) &&
            HAS_SOLUTION(x + 1, y + 1, z)) {
          out->push_back(make_tri(SOLUTION(x, y, z), NORMAL(x, y, z),
                                  SOLUTION(x, y + 1, z), NORMAL(x, y + 1, z),
                                  SOLUTION(x + 1, y, z), NORMAL(x + 1, y, z)));
          out->push_back(make_tri(SOLUTION(x + 1, y + 1, z),
                                  NORMAL(x + 1, y + 1, z),
                                  SOLUTION(x, y + 1, z), NORMAL(x, y + 1, z),
                                  SOLUTION(x + 1, y, z), NORMAL(x + 1, y, z)));
        }

        if (HAS_SOLUTION(x, y, z + 1) && HAS_SOLUTION(x + 1, y, z) &&
            HAS_SOLUTION(x + 1, y, z + 1)) {
          out->push_back(make_tri(SOLUTION(x, y, z), NORMAL(x, y, z),
                                  SOLUTION(x, y, z + 1), NORMAL(x, y, z + 1),
                                  SOLUTION(x + 1, y, z), NORMAL(x + 1, y, z)));
          out->push_back(make_tri(SOLUTION(x + 1, y, z + 1),
                                  NORMAL(x + 1, y, z + 1),
                                  SOLUTION(x, y, z + 1), NORMAL(x, y, z + 1),
                                  SOLUTION(x + 1, y, z), NORMAL(x + 1, y, z)));
        }

        if (HAS_SOLUTION(x, y, z + 1) && HAS_SOLUTION(x, y + 1, z) &&
            HAS_SOLUTION(x, y + 1, z + 1)) {
          out->push_back(make_tri(SOLUTION(x, y, z), NORMAL(x, y, z),
                                  SOLUTION(x, y, z + 1), NORMAL(x, y, z + 1),
                                  SOLUTION(x, y + 1, z), NORMAL(x, y + 1, z)));
          out->push_back(make_tri(SOLUTION(x, y + 1, z + 1),
                                  NORMAL(x, y + 1, z + 1),
                                  SOLUTION(x, y, z + 1), NORMAL(x, y, z + 1),
                                  SOLUTION(x, y + 1, z), NORMAL(x, y + 1, z)));
        }
      }
    }
  }

  free(solutions);
  free(inside_corners);

  return out;
}
