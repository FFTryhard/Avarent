#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <iostream>

float some_fn(float x, float y){
    return x + y;
}

PYBIND11_MODULE(module_name, handle){
    handle.doc() = "Poep";
    handle.def("some fn bs", &some_fn);
}