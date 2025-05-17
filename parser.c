#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "parser.h"

static const char key_word_png[] = "png";
static const char key_word_up[] = "up";
static const char key_word_eye[] = "eye";
static const char key_word_forward[] = "forward";
static const char key_word_sphere[] = "sphere";
static const char key_word_sun[] = "sun";
static const char key_word_color[] = "color";
static const char key_word_expose[] = "expose";
static const char key_word_panorama[] = "panorama";
static const char key_word_fisheye[] = "fisheye";
static const char key_word_plane[] = "plane";

bool prefix(const char *pre, const char *str) {
    return strncmp(pre, str, strlen(pre)) == 0;
}

void add_token(char ***dest, int *current_size, const char *new_token) {
    char **temp = realloc(*dest, (*current_size + 1) * sizeof(char *));

    if (temp == NULL) {
        printf("Memory allocation failed!\n");
        exit(1);
    }

    *dest = temp;

    (*dest)[*current_size] = malloc((strlen(new_token) + 1) * sizeof(char));

    if ((*dest)[*current_size] == NULL) {
        printf("Memory allocation for string failed!\n");
        exit(1);
    }

    strcpy((*dest)[*current_size], new_token);
    (*current_size)++;
}

void tokenize_line(char ***tokens, int *token_count, const char *line, const size_t start_pos, const int line_length) {
    int current_pos = start_pos;
    while (current_pos < line_length && line[current_pos] != '\n') {
        int non_space_length = 0;
        while (!isspace(line[current_pos]) && line[current_pos] != '\0') {
            non_space_length++;
            current_pos++;
        }

        if (non_space_length > 0) {
            char token[non_space_length + 1];
            int c = 0;
            while (c < non_space_length) {
                token[c] = line[current_pos - non_space_length + c];
                c++;
            }
            // terminate
            token[c] = '\0';

            add_token(tokens, token_count, token);
        } else {
            current_pos++;
        }
    }
}

void free_tokens(char **arr, const int size) {
    for (int i = 0; i < size; i++) {
        free(arr[i]);
    }
    free(arr);
}

bool string_is_number(const char *str) {
    for (int i = 0; i < strlen(str); i++) {
        if (!isdigit(str[i])) {
            return false;
        }
    }

    return true;
}

void copy_vector(double *dest, const int length, const char **src) {
    for (int i = 0; i < length; i++) {
        char *pEnd;
        dest[i] = strtod(src[i], &pEnd);
    }
}

void parse(FILE *file,
           char **filename,
           image_t **img,
           object *objects,
           object *suns) {

    char *line = NULL;
    size_t line_length = 0;

    double color[3] = {1.0, 1.0, 1.0};
    while (getline(&line, &line_length, file) != -1) {
        if (prefix(key_word_png, line)) {
            // may be an edge case where every character ends up in its own column in which this doesn't compensate
            // for the extra null terminator
            char **tokens = malloc(1 * sizeof(char *));
            int token_count = 0;
            tokenize_line(&tokens, &token_count, line, strlen(key_word_png), line_length);

            int dimensions[2];

            // it's a hack, don't judge me
            int num_count = 0;

            for (int i = 0; i < token_count; i++) {
                if (string_is_number(tokens[i])) {
                    dimensions[num_count] = atoi(tokens[i]);
                    num_count++;
                } else {
                    *filename = malloc(strlen(tokens[i]) + 1);
                    strcpy(*filename, tokens[i]);
                }
            }
            *img = new_image(dimensions[0], dimensions[1]);

            free_tokens(tokens, token_count);
        } else if (prefix(key_word_sphere, line)) {
            objects->vector_count++;
            char **tokens = malloc(1 * sizeof(char *));
            int token_count = 0;
            tokenize_line(&tokens, &token_count, line, strlen(key_word_sphere), line_length);

            double sphere[4];
            copy_vector(sphere, 4, tokens);

            objects->vectors[objects->vector_count - 1].x = sphere[0];
            objects->vectors[objects->vector_count - 1].y = sphere[1];
            objects->vectors[objects->vector_count - 1].z = sphere[2];
            objects->vectors[objects->vector_count - 1].radius = sphere[3];
            objects->vectors[objects->vector_count - 1].r = color[0];
            objects->vectors[objects->vector_count - 1].g = color[1];
            objects->vectors[objects->vector_count - 1].b = color[2];
            objects->vectors[objects->vector_count - 1].type = 0;

            free_tokens(tokens, token_count);
        } else if (prefix(key_word_sun, line)) {
            suns->vector_count++;
            char **tokens = malloc(1 * sizeof(char *));
            int token_count = 0;
            tokenize_line(&tokens, &token_count, line, strlen(key_word_sun), line_length);

            double sun[3];
            copy_vector(sun, 3, tokens);

            suns->vectors[suns->vector_count - 1].x = sun[0];
            suns->vectors[suns->vector_count - 1].y = sun[1];
            suns->vectors[suns->vector_count - 1].z = sun[2];
            suns->vectors[suns->vector_count - 1].radius = 0.0f;
            suns->vectors[suns->vector_count - 1].r = color[0];
            suns->vectors[suns->vector_count - 1].g = color[1];
            suns->vectors[suns->vector_count - 1].b = color[2];
            suns->vectors[suns->vector_count - 1].type = 1;

            free_tokens(tokens, token_count);
        } else if (prefix(key_word_plane, line)) {
            objects->vector_count++;
            char **tokens = malloc(1 * sizeof(char *));
            int token_count = 0;
            tokenize_line(&tokens, &token_count, line, strlen(key_word_plane), line_length);
            double temp[4];
            copy_vector(temp, 4, tokens);

            objects->vectors[objects->vector_count - 1].x = temp[0];
            objects->vectors[objects->vector_count - 1].y = temp[1];
            objects->vectors[objects->vector_count - 1].z = temp[2];
            objects->vectors[objects->vector_count - 1].radius = temp[3];
            objects->vectors[objects->vector_count - 1].r = color[0];
            objects->vectors[objects->vector_count - 1].g = color[1];
            objects->vectors[objects->vector_count - 1].b = color[2];
            objects->vectors[objects->vector_count - 1].type = 2;

            free_tokens(tokens, token_count);
        } else if (prefix(key_word_color, line)) {
            char **tokens = malloc(1 * sizeof(char *));
            int token_count = 0;
            tokenize_line(&tokens, &token_count, line, strlen(key_word_color), line_length);
            copy_vector(color, 3, tokens);
            free_tokens(tokens, token_count);
        } else if (prefix(key_word_up, line)) {
            char **tokens = malloc(1 * sizeof(char *));
            int token_count = 0;
            tokenize_line(&tokens, &token_count, line, strlen(key_word_up), line_length);
            copy_vector(up, 3, tokens);
            vec3_cross_product(up, forward, right);
            vec3_normalize(up, up);
            vec3_cross_product(right, forward, up);
            vec3_normalize(right, right);
            free_tokens(tokens, token_count);
        } else if (prefix(key_word_eye, line)) {
            char **tokens = malloc(1 * sizeof(char *));
            int token_count = 0;
            tokenize_line(&tokens, &token_count, line, strlen(key_word_eye), line_length);
            copy_vector(eye, 3, tokens);
            free_tokens(tokens, token_count);
        } else if (prefix(key_word_forward, line)) {
            char **tokens = malloc(1 * sizeof(char *));
            int token_count = 0;
            tokenize_line(&tokens, &token_count, line, strlen(key_word_forward), line_length);
            copy_vector(forward, 3, tokens);
            vec3_cross_product(up, forward, right);
            vec3_normalize(up, up);
            vec3_cross_product(right, forward, up);
            vec3_normalize(right, right);
            free_tokens(tokens, token_count);
        } else if (prefix(key_word_expose, line)) {
            char **tokens = malloc(1 * sizeof(char *));
            int token_count = 0;
            tokenize_line(&tokens, &token_count, line, strlen(key_word_expose), line_length);
            char *pEnd;
            exposure = strtod(tokens[0], &pEnd);
            exposed = true;
            free_tokens(tokens, token_count);
        } else if (prefix(key_word_panorama, line)) {
            panorama = true;
        } else if (prefix(key_word_fisheye, line)) {
            fisheye = true;
        }
    }
}
