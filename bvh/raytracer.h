#pragma once
#include "uselibpng.h"

extern double eye[3];
extern double forward[3];
extern double right[3];
extern double up[3];

extern bool exposed;
extern double exposure;

extern bool panorama;
extern bool fisheye;

extern bool dof;
extern double focus;
extern double lens;

typedef struct {
    double x, y, z;
    double r, g, b;
    double radius;
    int type;
} vec;

typedef struct {
    vec *vectors;
    int vector_count;
} object;

typedef struct {
    double a, b, c, d;
    // double red, green, blue;
} plane;

typedef struct {
    plane *planes;
    int plane_count;
} planes;

// Add these new structures to help manage materials and lighting
typedef struct {
    double diffuse[3];
    double reflectivity;
} material_t;

typedef struct {
    bool hit;
    double distance;
    double point[3];
    double normal[3];
} ray_result;

// New AABB structure
typedef struct {
    double min[3];  // Minimum point (a,b,c)
    double max[3];  // Maximum point (A,B,C)
} aabb_t;

// BVH node structure
typedef struct bvh_node {
    aabb_t bounds;
    struct bvh_node* left;
    struct bvh_node* right;
    vec* objects;
    int object_count;
} bvh_node_t;

typedef struct {
    bvh_node_t* node;
    double t_min;
} stack_entry;

void free_object(object *obj);

void free_planes(planes *planes);

void vec3_normalize(double *result, const double *v);

void vec3_cross_product(double res[3], double a[3], double b[3]);

// void draw(image_t *img, const object *spheres, const object *suns, const planes *planes);
void draw(image_t *img, const object *objects, const object *suns);
