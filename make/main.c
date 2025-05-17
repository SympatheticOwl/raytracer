#include <stdio.h>
#include <stdlib.h>

#include "raytracer.h"
#include "parser.h"

int main(int argc, const char **argv) {
    if (argc != 2) {
        perror("Usage: main file=...\n");
        return 1;
    }
    int file_name_index = argc - 1;
    const char *file_name = argv[file_name_index];
    printf("file name: %s\n", file_name);

    FILE *file = fopen(file_name, "r");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    char *filename = NULL;

    image_t *img;

    //TODO: realloc for many.txt
    object *objects = calloc(1, sizeof(objects));
    objects->vector_count = 0;
    objects->vectors = calloc(2048, sizeof(objects));
    object *suns = calloc(1, sizeof(objects));
    suns->vector_count = 0;
    suns->vectors = calloc(2048, sizeof(objects));

    parse(file, &filename, &img, objects, suns);

    draw(img, objects, suns);

    fclose(file);

    save_image(img, filename);

    free_object(objects);
    free_object(suns);
    free(filename);
    free_image(img);

    return 0;
}
