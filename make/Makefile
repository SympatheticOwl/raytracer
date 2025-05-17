CC=clang
INCLUDES = -I. -I/usr/local/include
CFLAGS = -O3 $(INCLUDES)
LDFLAGS = -L/usr/local/lib

OBJS = main.o raytracer.o parser.o uselibpng.o

.PHONY: build, run


run: build
	./main $(file)

build: $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -lpng -lm $(OBJS) -o main

main.o: main.c raytracer.h parser.h uselibpng.h
	$(CC) $(CFLAGS) -c main.c

raytracer.o: raytracer.c raytracer.h uselibpng.h
	$(CC) $(CFLAGS) -c raytracer.c

parser.o: parser.c parser.h raytracer.h
	$(CC) $(CFLAGS) -c parser.c

uselibpng.o: uselibpng.c uselibpng.h
	$(CC) $(CFLAGS) -c uselibpng.c

clean:
	rm -rf main *.o *.png

