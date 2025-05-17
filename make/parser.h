#pragma once
#include "raytracer.h"

void parse(FILE *file,
           char **filename,
           image_t **img,
           object *objects,
           object *suns);