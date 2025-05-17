#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>
#include "raytracer.h"

#define EPSILON 0.0001f

double eye[3] = {0, 0, 0};
double forward[3] = {0, 0, -1};
double right[3] = {1, 0, 0};
double up[3] = {0, 1, 0};

bool exposed = false;
double exposure = 0.0f;

bool fisheye = false;
bool panorama = false;

void free_object(object *obj) {
    free(obj->vectors);
    free(obj);
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

void vec3_cross_product(double res[3], const double a[3], const double b[3]) {
    res[0] = a[1] * b[2] - a[2] * b[1];
    res[1] = a[2] * b[0] - a[0] * b[2];
    res[2] = a[0] * b[1] - a[1] * b[0];
}

// TODO: theres something sometimes wrong with this
void apply_exposure(double *color) {
    if (!exposed) {
        return;
    }

    for (int i = 0; i < 3; i++) {
        color[i] = 1.0f - expf(-exposure * color[i]);
    }
}

void ray_sphere_intersect(ray_result *res,
                          double ray[2][3],
                          double sphere[4]) {
    const double radius = sphere[3];
    const double *ro = ray[0]; // origin
    const double *rd = ray[1]; // direction

    double temp[3];
    vec3_subtract(temp, sphere, ro);
    const double c_minus_ro_squared = vec3_squared_length(temp);
    const double radius_squared = radius * radius;
    const bool inside = c_minus_ro_squared < radius_squared;

    const double tc = vec3_dot(temp, rd); // rd is already normalized (unit-length)

    if (!inside && tc < 0) {
        return;
    }

    double closest_point[3];
    vec3_scale_add(closest_point, ro, rd, tc);
    vec3_subtract(temp, closest_point, sphere);
    const double d_squared = vec3_squared_length(temp);

    if (!inside && radius_squared < d_squared) {
        return;
    }

    const double t_offset = sqrtf(radius_squared - d_squared); // rd is unit length, so no division needed

    double t;
    if (inside) {
        t = tc + t_offset;
    } else {
        t = tc - t_offset;
    }

    res->hit = true;
    res->distance = t;
    vec3_scale_add(res->point, ro, rd, t);

    vec3_subtract(temp, res->point, sphere);
    vec3_normalize(res->normal, temp);
}

void ray_plane_intersect(ray_result *res,
                        double ray[2][3],
                        const double plane[4]) {
    const double *ro = ray[0];  // origin
    const double *rd = ray[1];  // direction

    double normal[3] = {plane[0], plane[1], plane[2]};
    double d = plane[3];

    double denom = vec3_dot(rd, normal);

    if (fabs(denom) < EPSILON) {
        return;
    }

    const double point_on_plane[3] = {-d * plane[0], -d * plane[1], -d * plane[2]};

    double temp[3];
    vec3_subtract(temp, point_on_plane, ro);

    const double t = vec3_dot(temp, normal) / denom;

    if (t < EPSILON) {
        return;
    }

    res->hit = true;
    res->distance = t;

    vec3_scale_add(res->point, ro, rd, t);

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

void ray_intersection(ray_result *res,
                      double ray[2][3],
                      const vec *v) {
    const double obj[4] = {v->x, v->y, v->z, v->radius};
    if (v->type == 0) {
        ray_sphere_intersect(res, ray, obj);
    } else if (v->type == 2) {
        ray_plane_intersect(res, ray, obj);
    }
}

void spherical_to_cartesian(double result[3], const double latitude, const double longitude) {
    result[0] = cosf(latitude) * sinf(longitude);
    result[1] = sinf(latitude);
    result[2] = -cosf(latitude) * cosf(longitude);
}

void create_camera_ray(double ray[2][3], const int x, const int y, const int width, const int height) {
    if (panorama) {
        double longitude = ((double)x / width) * 2.0f * M_PI - M_PI;
        double latitude = ((double)(height - y) / height) * M_PI - M_PI/2;

        double direction[3];
        spherical_to_cartesian(direction, latitude, longitude);

        double final_direction[3] = {0, 0, 0};

        double cam_matrix[3][3] = {
            {right[0],   up[0],   -forward[0]},
            {right[1],   up[1],   -forward[1]},
            {right[2],   up[2],   -forward[2]}
        };

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
        const double sx = (2.0f * x - width) / fmax(width, height);
        const double sy = (height - 2.0f * y) / fmax(width, height);

        if (sx * sx + sy * sy > 1.0f) {
            ray[0][0] = eye[0];
            ray[0][1] = eye[1];
            ray[0][2] = eye[2];
            ray[1][0] = 0;
            ray[1][1] = 0;
            ray[1][2] = 0;
            return;
        }

        double scaled_right[3], scaled_up[3], scaled_forward[3];
        double direction[3];
        double temp[3];

        vec3_scale(scaled_right, right, sx);
        vec3_scale(scaled_up, up, sy);

        double forward_scale = sqrtf(1.0f - (sx * sx + sy * sy));
        vec3_scale(scaled_forward, forward, forward_scale);

        vec3_add(temp, scaled_right, scaled_up);
        vec3_add(direction, temp, scaled_forward);

        ray[0][0] = eye[0];
        ray[0][1] = eye[1];
        ray[0][2] = eye[2];
        ray[1][0] = direction[0];
        ray[1][1] = direction[1];
        ray[1][2] = direction[2];
    } else {
        double sx = (2.0f * x - width) / fmax(width, height);
        double sy = (height - 2.0f * y) / fmax(width, height);

        double scaled_right[3], scaled_up[3];

        // f + sx*r + sy*u
        double direction[3];
        vec3_scale(scaled_right, right, sx);
        vec3_scale(scaled_up, up, sy);
        double temp[3];
        vec3_add(temp, forward, scaled_right);
        vec3_add(direction, temp, scaled_up);

        vec3_normalize(direction, direction);

        ray[0][0] = eye[0];
        ray[0][1] = eye[1];
        ray[0][2] = eye[2];
        ray[1][0] = direction[0];
        ray[1][1] = direction[1];
        ray[1][2] = direction[2];
    }
}

double clamp(const double value) {
    if (value < 0.0f) return 0.0f;
    if (value > 1.0f) return 1.0f;
    return value;
}

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

        ray_intersection(&shadow_res, ray, &objects->vectors[i]);

        if (shadow_res.hit && shadow_res.distance > EPSILON && shadow_res.distance < max_distance) {
            return true; // Point is in shadow
        }
    }
    return false;
}

void draw(image_t *img, const object *objects, const object *suns) {
    if (objects->vector_count == 0) {
        return;
    }

    const time_t start_time = time(NULL);
    const clock_t start_cpu = clock();

    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            double ray[2][3] = {{0, 0, 0}, {0, 0, 0}};
            create_camera_ray(ray, x, y, img->width, img->height);

            double smallest_t = INFINITY;
            bool hit_anything = false;
            ray_result closest_hit;
            int closest_object_index = -1;

            for (int i = 0; i < objects->vector_count; i++) {
                ray_result res;
                res.hit = false;
                res.distance = 0.0;
                res.point[0] = 0.0;
                res.point[1] = 0.0;
                res.point[2] = 0.0;

                const vec v = objects->vectors[i];
                ray_intersection(&res, ray, &v);

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
                final_color[0] = clamp(lit_color[0]);
                final_color[1] = clamp(lit_color[1]);
                final_color[2] = clamp(lit_color[2]);

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

    const time_t end_time = time(NULL);
    const clock_t end_cpu = clock();

    const double cpu_time_used = (double) (end_cpu - start_cpu) / CLOCKS_PER_SEC;
    const double wall_time_used = difftime(end_time, start_time);

    printf("Render completed:\n");
    printf("  Wall clock time: %.2f seconds\n", wall_time_used);
    printf("  CPU time used: %.2f seconds\n", cpu_time_used);
}
