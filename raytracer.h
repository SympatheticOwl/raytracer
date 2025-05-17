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

void free_object(object *obj);

void vec3_normalize(double *result, const double *v);

void vec3_cross_product(double res[3], const double a[3], const double b[3]);

void draw(image_t *img, const object *objects, const object *suns);
