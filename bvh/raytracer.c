#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#include "raytracer.h"

#include <stdio.h>

double eye[3] = {0, 0, 0};
double forward[3] = {0, 0, -1};
double right[3] = {1, 0, 0};
double up[3] = {0, 1, 0};

bool exposed = false;
double exposure = 0.0f;

bool fisheye = false;
bool panorama = false;

bool dof = false;
double focus = 0.0f;
double lens = 0.0f;

#define EPSILON 0.0001f

void free_object(object *obj) {
    free(obj->vectors);
    free(obj);
}

void free_planes(planes *planes) {
    free(planes->planes);
    free(planes);
}

double linear_to_srgb(double linear) {
    if (linear <= 0.0031308) {
        return linear * 12.92;
    }
    return 1.055 * pow(linear, 1.0 / 2.4) - 0.055;
}

void convert_to_srgb(double *color) {
    for (int i = 0; i < 3; i++) {
        color[i] = linear_to_srgb(color[i]);
    }
}

void vec3_add(double *result, const double *a, const double *b) {
    result[0] = a[0] + b[0];
    result[1] = a[1] + b[1];
    result[2] = a[2] + b[2];
}

void vec3_scale(double *result, const double *v, double s) {
    result[0] = v[0] * s;
    result[1] = v[1] * s;
    result[2] = v[2] * s;
}

double vec3_length(const double *v) {
    return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

void vec3_normalize(double *result, const double *v) {
    const double length = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

    result[0] = v[0] / length;
    result[1] = v[1] / length;
    result[2] = v[2] / length;
}

double vec3_dot(const double *a, const double *b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double vec3_squared_length(const double *v) {
    return vec3_dot(v, v);
}

void vec3_subtract(double *result, const double *a, const double *b) {
    result[0] = a[0] - b[0];
    result[1] = a[1] - b[1];
    result[2] = a[2] - b[2];
}

void vec3_scale_add(double *result, const double *a, const double *dir, double t) {
    result[0] = a[0] + dir[0] * t;
    result[1] = a[1] + dir[1] * t;
    result[2] = a[2] + dir[2] * t;
}

void vec3_cross_product(double res[3], double a[3], double b[3]) {
    res[0] = a[1] * b[2] - a[2] * b[1];
    res[1] = a[2] * b[0] - a[0] * b[2];
    res[2] = a[0] * b[1] - a[1] * b[0];
}

void apply_exposure(double *color) {
    if (!exposed) {
        // If exposure is disabled, color remains linear
        return;
    }

    // Apply exposure function: 1 - e^(-v * linear)
    // where v is exposure_value and linear is the input color
    for (int i = 0; i < 3; i++) {
        color[i] = 1.0f - expf(-exposure * color[i]);
    }
}

void ray_sphere_intersect(ray_result *res,
                          double ray[2][3],
                          double sphere[4]) {
    const double radius = sphere[3];
    const double *ro = ray[0];
    const double *rd = ray[1];

    double temp[3];
    vec3_subtract(temp, sphere, ro);
    const double c_minus_ro_squared = vec3_squared_length(temp);
    const double radius_squared = radius * radius;
    const bool inside = c_minus_ro_squared < radius_squared;

    // 2. Calculate tc (closest approach)
    const double tc = vec3_dot(temp, rd); // rd is already normalized (unit-length)

    // 3. Early exit check
    if (!inside && tc < 0) {
        return;
    }

    // 4. Calculate squared distance at closest approach
    double closest_point[3];
    vec3_scale_add(closest_point, ro, rd, tc);
    vec3_subtract(temp, closest_point, sphere);
    const double d_squared = vec3_squared_length(temp);

    // 5. Check if ray misses sphere
    if (!inside && radius_squared < d_squared) {
        return;
    }

    // 6. Calculate offset from closest approach to intersection point
    const double t_offset = sqrtf(radius_squared - d_squared); // rd is unit length, so no division needed

    // 7. Calculate intersection point and distance
    double t;
    if (inside) {
        t = tc + t_offset; // Ray originates inside sphere, so use far intersection
    } else {
        t = tc - t_offset; // Ray originates outside sphere, so use near intersection
    }

    // Set result values
    res->hit = true;
    res->distance = t;
    vec3_scale_add(res->point, ro, rd, t);

    vec3_subtract(temp, res->point, sphere);
    vec3_normalize(res->normal, temp);
}

// Add ray-plane intersection function
void ray_plane_intersect(ray_result *res,
                        double ray[2][3],
                        const double plane[4]) {
    // Create temporary plane structure from vec
    const double *ro = ray[0];  // Ray origin
    const double *rd = ray[1];  // Ray direction

    // Normal is stored in x,y,z components
    double normal[3] = {plane[0], plane[1], plane[2]};
    double d = plane[3];  // Using radius field to store d component

    // Calculate denominator (ray direction dot normal)
    double denom = vec3_dot(rd, normal);

    // If denominator is near zero, ray is parallel to plane
    if (fabs(denom) < EPSILON) {
        return;  // No intersection
    }

    // Calculate point on plane using d component
    double point_on_plane[3] = {-d * plane[0], -d * plane[1], -d * plane[2]};

    // Calculate vector from ray origin to point on plane
    double temp[3];
    vec3_subtract(temp, point_on_plane, ro);

    // Calculate intersection distance
    double t = vec3_dot(temp, normal) / denom;

    // If t is negative, intersection is behind ray origin
    if (t < EPSILON) {
        return;
    }

    // We have a valid intersection
    res->hit = true;
    res->distance = t;

    // Calculate intersection point
    vec3_scale_add(res->point, ro, rd, t);

    // Set normal (flip if necessary to face ray origin)
    if (denom > 0) {
        res->normal[0] = -normal[0];
        res->normal[1] = -normal[1];
        res->normal[2] = -normal[2];
    } else {
        res->normal[0] = normal[0];
        res->normal[1] = normal[1];
        res->normal[2] = normal[2];
    }
}

void ray_primitive_intersect(ray_result *res,
                             double ray[2][3],
                             const vec *v) {
    const double obj[4] = {v->x, v->y, v->z, v->radius};
    if (v->type == 0) {
        ray_sphere_intersect(res, ray, obj);
    } else if (v->type == 2) {
        ray_plane_intersect(res, ray, obj);
    }
}

// Add these helper functions for spherical coordinates
void spherical_to_cartesian(double latitude, double longitude, double result[3]) {
    // Convert latitude/longitude to cartesian coordinates
    // latitude: -π/2 to π/2 (from south pole to north pole)
    // longitude: -π to π (around the equator)
    result[0] = cosf(latitude) * sinf(longitude);  // x
    result[1] = sinf(latitude);                    // y (up)
    result[2] = -cosf(latitude) * cosf(longitude); // z (forward is negative z)
}

// Improved random disk sampling
void random_point_on_disk(double radius, double result[3], const double normal[3]) {
    // Use rejection sampling for better distribution
    double x, y;
    do {
        x = 2.0f * ((double)rand() / RAND_MAX) - 1.0f;  // Range [-1, 1]
        y = 2.0f * ((double)rand() / RAND_MAX) - 1.0f;  // Range [-1, 1]
    } while (x*x + y*y > 1.0f);  // Reject points outside unit circle

    // Scale by radius
    x *= radius;
    y *= radius;

    // Create orthonormal basis (u, v, normal)
    double u[3], v[3];

    // Find first perpendicular vector
    if (fabsf(normal[0]) < fabsf(normal[1])) {
        vec3_cross_product(u, (double[3]){1, 0, 0}, normal);
    } else {
        vec3_cross_product(u, (double[3]){0, 1, 0}, normal);
    }
    vec3_normalize(u, u);

    // Find second perpendicular vector
    vec3_cross_product(v, normal, u);
    vec3_normalize(v, v);

    // Compute point on disk
    double scaled_u[3], scaled_v[3];
    vec3_scale(scaled_u, u, x);
    vec3_scale(scaled_v, v, y);

    vec3_add(result, scaled_u, scaled_v);
}

void apply_dof(double ray[2][3]) {

    // Store original ray origin and direction
    double original_origin[3] = {ray[0][0], ray[0][1], ray[0][2]};
    double original_dir[3] = {ray[1][0], ray[1][1], ray[1][2]};

    // Calculate point on focal plane
    double focal_point[3];
    vec3_scale_add(focal_point, original_origin, original_dir, focus);

    // Generate random point on lens
    double lens_offset[3];
    random_point_on_disk(lens, lens_offset, forward);

    // Calculate new origin by adding lens offset to eye position
    double new_origin[3];
    vec3_add(new_origin, original_origin, lens_offset);

    // Calculate new direction from new origin to focal point
    double new_direction[3];
    vec3_subtract(new_direction, focal_point, new_origin);
    vec3_normalize(new_direction, new_direction);

    // Update ray with new values
    ray[0][0] = new_origin[0];
    ray[0][1] = new_origin[1];
    ray[0][2] = new_origin[2];
    ray[1][0] = new_direction[0];
    ray[1][1] = new_direction[1];
    ray[1][2] = new_direction[2];
}

void create_camera_ray(double ray[2][3], int x, int y, int width, int height) {
    if (panorama) {
        // Panoramic projection
        // Map x to longitude (-π to π) and y to latitude (-π/2 to π/2)
        double longitude = ((double)x / width) * 2.0f * M_PI - M_PI;    // Range: -π to π
        double latitude = ((double)(height - y) / height) * M_PI - M_PI/2; // Range: -π/2 to π/2

        // Calculate direction vector from latitude/longitude
        double direction[3];
        spherical_to_cartesian(latitude, longitude, direction);

        // Transform direction to camera space
        double final_direction[3] = {0, 0, 0};

        // Create camera transformation matrix
        double cam_matrix[3][3] = {
            {right[0],   up[0],   -forward[0]},
            {right[1],   up[1],   -forward[1]},
            {right[2],   up[2],   -forward[2]}
        };

        // Apply camera transformation
        for (int i = 0; i < 3; i++) {
            final_direction[i] =
                cam_matrix[i][0] * direction[0] +
                cam_matrix[i][1] * direction[1] +
                cam_matrix[i][2] * direction[2];
        }

        ray[0][0] = eye[0];
        ray[0][1] = eye[1];
        ray[0][2] = eye[2];
        ray[1][0] = final_direction[0];
        ray[1][1] = final_direction[1];
        ray[1][2] = final_direction[2];
    } else if (fisheye) {
        // Fisheye projection
        double sx = (2.0f * x - width) / (double)fmax(width, height);
        double sy = (height - 2.0f * y) / (double)fmax(width, height);

        // Check if point is within the fisheye circle
        if (sx * sx + sy * sy > 1.0f) {
            // Point is outside the fisheye view, set hit to false
            ray[0][0] = eye[0];
            ray[0][1] = eye[1];
            ray[0][2] = eye[2];
            ray[1][0] = 0;
            ray[1][1] = 0;
            ray[1][2] = 0;
            return;
        }

        // Calculate the direction using the suggested formula
        double scaled_right[3], scaled_up[3], scaled_forward[3];
        double direction[3];
        double temp[3];

        // Scale the basis vectors
        vec3_scale(scaled_right, right, sx);
        vec3_scale(scaled_up, up, sy);

        // Calculate sqrt(1 - sx^2 - sy^2) * forward
        double forward_scale = sqrtf(1.0f - (sx * sx + sy * sy));
        vec3_scale(scaled_forward, forward, forward_scale);

        // Combine the vectors
        vec3_add(temp, scaled_right, scaled_up);
        vec3_add(direction, temp, scaled_forward);

        // The direction is already normalized by construction

        ray[0][0] = eye[0];
        ray[0][1] = eye[1];
        ray[0][2] = eye[2];
        ray[1][0] = direction[0];
        ray[1][1] = direction[1];
        ray[1][2] = direction[2];
    } else {
        double sx = (2.0f * x - width) / (double) fmax(width, height);
        double sy = (height - 2.0f * y) / (double) fmax(width, height);

        double scaled_right[3], scaled_up[3];

        // Calculate direction: f + sx*r + sy*u
        double direction[3];
        vec3_scale(scaled_right, right, sx);
        vec3_scale(scaled_up, up, sy);
        double temp[3];
        vec3_add(temp, forward, scaled_right);
        vec3_add(direction, temp, scaled_up);

        // Normalize the direction vector
        vec3_normalize(direction, direction);

        ray[0][0] = eye[0];
        ray[0][1] = eye[1];
        ray[0][2] = eye[2];
        ray[1][0] = direction[0];
        ray[1][1] = direction[1];
        ray[1][2] = direction[2];

        if (dof) {
            apply_dof(ray);
        }
    }
}

// Clamp value to [0,1] range
double clamp01(double value) {
    if (value < 0.0f) return 0.0f;
    if (value > 1.0f) return 1.0f;
    return value;
}

// Add shadow testing function
bool test_shadow(const double origin[3],
                 const double direction[3],
                 double max_distance,
                 const object *objects) {
    double ray[2][3] = {
        {origin[0], origin[1], origin[2]},
        {direction[0], direction[1], direction[2]}
    };

    for (int i = 0; i < objects->vector_count; i++) {
        ray_result shadow_res;
        shadow_res.hit = false;

        ray_primitive_intersect(&shadow_res, ray, &objects->vectors[i]);

        if (shadow_res.hit && shadow_res.distance > EPSILON && shadow_res.distance < max_distance) {
            return true; // Point is in shadow
        }
    }
    return false;
}

// Ray-AABB intersection test based on the documentation
bool ray_aabb_intersect(const double ray[2][3], const aabb_t* aabb) {
    double tx1 = (aabb->min[0] - ray[0][0]) / ray[1][0];
    double tx2 = (aabb->max[0] - ray[0][0]) / ray[1][0];

    double tmin = fmin(tx1, tx2);
    double tmax = fmax(tx1, tx2);

    double ty1 = (aabb->min[1] - ray[0][1]) / ray[1][1];
    double ty2 = (aabb->max[1] - ray[0][1]) / ray[1][1];

    tmin = fmax(tmin, fmin(ty1, ty2));
    tmax = fmin(tmax, fmax(ty1, ty2));

    double tz1 = (aabb->min[2] - ray[0][2]) / ray[1][2];
    double tz2 = (aabb->max[2] - ray[0][2]) / ray[1][2];

    tmin = fmax(tmin, fmin(tz1, tz2));
    tmax = fmin(tmax, fmax(tz1, tz2));

    return tmax >= tmin && tmax > 0;
}

// Calculate AABB for a primitive
void calculate_primitive_aabb(const vec* primitive, aabb_t* aabb) {
    if (primitive->type == 0) {  // Sphere
        // For sphere, extend box by radius in all directions
        for (int i = 0; i < 3; i++) {
            aabb->min[i] = *(&primitive->x + i) - primitive->radius;
            aabb->max[i] = *(&primitive->x + i) + primitive->radius;
        }
    } else if (primitive->type == 2) {  // Plane
        // For plane, use a large box in the plane's direction
        // This is a simplified approach - could be optimized further
        const double LARGE_VALUE = 1000000.0;
        for (int i = 0; i < 3; i++) {
            if (fabs(*(&primitive->x + i)) > 0.1) {  // If normal component is significant
                aabb->min[i] = -LARGE_VALUE;
                aabb->max[i] = LARGE_VALUE;
            } else {
                aabb->min[i] = -LARGE_VALUE * 0.1;
                aabb->max[i] = LARGE_VALUE * 0.1;
            }
        }
    }
}

// Combine two AABBs
void combine_aabbs(const aabb_t* a, const aabb_t* b, aabb_t* result) {
    for (int i = 0; i < 3; i++) {
        result->min[i] = fmin(a->min[i], b->min[i]);
        result->max[i] = fmax(a->max[i], b->max[i]);
    }
}

// Build BVH node
bvh_node_t* build_bvh(vec* objects, int count) {
    bvh_node_t* node = malloc(sizeof(bvh_node_t));

    // Base case: leaf node
    if (count <= 4) {  // Small number of objects per leaf
        node->objects = objects;
        node->object_count = count;
        node->left = node->right = NULL;

        // Calculate bounds for all objects
        calculate_primitive_aabb(&objects[0], &node->bounds);
        for (int i = 1; i < count; i++) {
            aabb_t obj_bounds;
            calculate_primitive_aabb(&objects[i], &obj_bounds);
            combine_aabbs(&node->bounds, &obj_bounds, &node->bounds);
        }
        return node;
    }

    // Find longest axis of bounding box
    aabb_t bounds;
    calculate_primitive_aabb(&objects[0], &bounds);
    for (int i = 1; i < count; i++) {
        aabb_t obj_bounds;
        calculate_primitive_aabb(&objects[i], &obj_bounds);
        combine_aabbs(&bounds, &obj_bounds, &bounds);
    }

    // Find longest axis
    int axis = 0;
    double max_length = bounds.max[0] - bounds.min[0];
    for (int i = 1; i < 3; i++) {
        double length = bounds.max[i] - bounds.min[i];
        if (length > max_length) {
            max_length = length;
            axis = i;
        }
    }

    // Sort objects by centroid along longest axis
    double mid = (bounds.min[axis] + bounds.max[axis]) * 0.5;
    int split = 0;
    for (int i = 0; i < count; i++) {
        double center = *(&objects[i].x + axis);
        if (center < mid) {
            vec temp = objects[split];
            objects[split] = objects[i];
            objects[i] = temp;
            split++;
        }
    }

    // Ensure we don't create empty nodes
    if (split == 0 || split == count) {
        split = count / 2;
    }

    // Recursively build children
    node->left = build_bvh(objects, split);
    node->right = build_bvh(objects + split, count - split);
    node->objects = NULL;
    node->object_count = 0;

    // Combine children's bounds
    combine_aabbs(&node->left->bounds, &node->right->bounds, &node->bounds);

    return node;
}

void free_bvh(bvh_node_t* node) {
    if (node == NULL) {
        return;
    }

    // Recursively free children first
    if (node->left) {
        free_bvh(node->left);
    }
    if (node->right) {
        free_bvh(node->right);
    }

    // Note: We don't free node->objects because it points to the original objects array
    // which is managed by the caller

    // Free the node itself
    free(node);
}

// Add a helper to calculate maximum possible BVH depth
int calculate_max_bvh_depth(int object_count) {
    // For a binary tree, worst-case depth is ceil(log2(N))
    // Add some padding for safety
    return (int)(ceil(log2(object_count)) + 1);
}

void draw(image_t *img, const object *objects, const object *suns) {
    if (objects->vector_count == 0) {
        return;  // Early exit for empty scenes
    }

    // bvh_node_t* root = build_bvh(objects->vectors, objects->vector_count);

    // Calculate stack size based on maximum possible tree depth
    // int max_depth = calculate_max_bvh_depth(objects->vector_count);
    // int stack_capacity = max_depth * 2;  // Multiple by 2 for safety margin

    // Allocate stack dynamically
    // stack_entry *stack = (stack_entry*)malloc(stack_capacity * sizeof(stack_entry));

    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            double ray[2][3] = {{0, 0, 0}, {0, 0, 0}};
            create_camera_ray(ray, x, y, img->width, img->height);

            double smallest_t = INFINITY;
            bool hit_anything = false;
            ray_result closest_hit;
            int closest_object_index = -1;

            // Initialize stack for this ray
            int stack_size = 0;

            // // Push root node if ray intersects root bounds
            // if (ray_aabb_intersect(ray, &root->bounds)) {
            //     stack[stack_size++] = (stack_entry){root, 0};
            // }

            // while (stack_size > 0 && stack_size < stack_capacity) {
            //     stack_entry current = stack[--stack_size];
            //     bvh_node_t* node = current.node;
            //
            //     if (node->objects) {
            //         // Process leaf node
            //         for (int i = 0; i < node->object_count; i++) {
            //             ray_result res = {0};
            //             ray_primitive_intersect(&res, ray, &node->objects[i]);
            //
            //             if (res.hit && res.distance < smallest_t) {
            //                 smallest_t = res.distance;
            //                 closest_hit = res;
            //                 closest_object_index = node->objects - objects->vectors + i;
            //                 hit_anything = true;
            //             }
            //         }
            //     } else {
            //         // Process internal node, with stack overflow protection
            //         if (stack_size + 2 <= stack_capacity) {  // Ensure space for both children
            //             if (node->right && ray_aabb_intersect(ray, &node->right->bounds)) {
            //                 stack[stack_size++] = (stack_entry){node->right, 0};
            //             }
            //             if (node->left && ray_aabb_intersect(ray, &node->left->bounds)) {
            //                 stack[stack_size++] = (stack_entry){node->left, 0};
            //             }
            //         } else {
            //             fprintf(stderr, "Warning: BVH stack capacity exceeded at pixel (%d,%d)\n", x, y);
            //             break;  // Exit traversal loop if stack is full
            //         }
            //     }
            // }


            for (int i = 0; i < objects->vector_count; i++) {
                ray_result res;
                res.hit = false;
                res.distance = 0.0;
                res.point[0] = 0.0;
                res.point[1] = 0.0;
                res.point[2] = 0.0;

                const vec v = objects->vectors[i];
                ray_primitive_intersect(&res, ray, &v);

                if (res.hit && res.distance < smallest_t) {
                    smallest_t = res.distance;
                    closest_hit = res;
                    closest_object_index = i;
                    hit_anything = true;
                }
            }

            // Only process lighting if we hit something
            if (hit_anything) {
                const vec closest_object = objects->vectors[closest_object_index];
                double surface_color[3] = {closest_object.r, closest_object.g, closest_object.b};
                double lit_color[3] = {0, 0, 0};

                // Make object two-sided by checking if normal points away from ray
                double view_dot_normal = vec3_dot(ray[1], closest_hit.normal);
                if (view_dot_normal > 0) {
                    closest_hit.normal[0] = -closest_hit.normal[0];
                    closest_hit.normal[1] = -closest_hit.normal[1];
                    closest_hit.normal[2] = -closest_hit.normal[2];
                }

                // For each light source
                for (int j = 0; j < suns->vector_count; j++) {
                    vec sun = suns->vectors[j];
                    double light_dir[3] = {sun.x, sun.y, sun.z};
                    double light_color[3] = {sun.r, sun.g, sun.b};

                    // Normalize the light direction
                    vec3_normalize(light_dir, light_dir);

                    // Calculate Lambert's cosine law
                    double normal_dot_light = vec3_dot(closest_hit.normal, light_dir);

                    if (normal_dot_light > 0) {
                        // Create shadow ray origin with bias
                        double shadow_origin[3];
                        double bias_normal[3];
                        vec3_scale(bias_normal, closest_hit.normal, EPSILON);
                        vec3_add(shadow_origin, closest_hit.point, bias_normal);

                        // Check if point is in shadow
                        bool in_shadow = test_shadow(shadow_origin, light_dir, INFINITY, objects);

                        if (!in_shadow) {
                            lit_color[0] += surface_color[0] * light_color[0] * normal_dot_light;
                            lit_color[1] += surface_color[1] * light_color[1] * normal_dot_light;
                            lit_color[2] += surface_color[2] * light_color[2] * normal_dot_light;
                        }
                    }
                }

                // Clamp final colors to [0,1] range
                double final_color[3];
                final_color[0] = clamp01(lit_color[0]);
                final_color[1] = clamp01(lit_color[1]);
                final_color[2] = clamp01(lit_color[2]);

                // Apply exposure function
                apply_exposure(final_color);

                convert_to_srgb(final_color);

                // Only write pixel if we hit something
                pixel_xy(img, x, y).red = final_color[0] * 255;
                pixel_xy(img, x, y).green = final_color[1] * 255;
                pixel_xy(img, x, y).blue = final_color[2] * 255;
                pixel_xy(img, x, y).alpha = 255;
            }
        }
    }

    // free_bvh(root);
}
