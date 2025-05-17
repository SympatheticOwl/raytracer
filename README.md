# UIUC CS418 Raytracer

* This is an ok raytracer. I'd like to come back to it one day and make it realtime.
* There are 2 copies of the raytracer. The C files in the base folder and CMakeLists.txt were developed and heavily rely on CLion from JetBrains.
* The copy in the `make` folder should work on any system clang and has its own Makefile that can be run through the terminal agnostic of any Jetbrains requirements.
* Requires `libpng` be installed somewhere. Mine was installed at the system level on Apple Silicon so the Clion files are heavily tied to that but the `make` directory was tested on linux.
* The `bvh` was an attempt at a bounding volume hierarchy. It kinda worked but not well enough to submit it.