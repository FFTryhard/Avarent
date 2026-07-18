#include "python_runtime.h"
#include <pybind11/embed.h>
#include <Windows.h>

#include <iostream>

namespace py = pybind11;

#include <windows.h>

void init_python()
{

    Py_SetPythonHome(
        L"C:\\Users\\H\\AppData\\Local\\Programs\\Python\\Python312"
    );

    static py::scoped_interpreter guard{};

    try {
        py::exec(R"(
            import os
            import sys
            os.add_dll_directory(r"C:\Users\H\AppData\Local\Programs\Python\Python312")
            os.add_dll_directory(r"C:\Users\H\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy.libs")
            os.add_dll_directory(r"C:\Users\H\AppData\Local\Programs\Python\Python312\Lib\site-packages\scipy.libs")
            print("Python runtime init: prefix=", sys.prefix)
        )");
    } catch (const py::error_already_set& e) {
        std::cerr << "Python initialization error: " << e.what() << std::endl;
    }
}