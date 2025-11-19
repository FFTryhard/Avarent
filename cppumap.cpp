#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include <random>
#include <filesystem>
#include "utilspy.h"
#include "cppumap.h"

namespace py = pybind11;

std::vector<std::vector<double>> UMAP(std::vector<std::vector<double>> data_target){

    std::cout << "Entered Main.\n";

    Py_SetPythonHome(L"C:/Users/H/AppData/Local/Programs/Python/Python312");

    // 2. (optional) Change working directory if your Python scripts live there
    std::filesystem::current_path("C:/Users/H/Desktop/cpp/MEEM");

    // 3. Start the interpreter
    py::scoped_interpreter guard{};
    std::cout << "Interpreter initialized successfully.\n";

    // 4. Make sure site-packages is visible
    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("insert")(0, "C:/Users/H/AppData/Local/Programs/Python/Python312/Lib/site-packages");

    // Debug print
    for (auto item : sys.attr("path")) {
        std::cout << "PY PATH: " << py::str(item).cast<std::string>() << std::endl;
    }

    // 5. Test import
    try {
        std::cout << "Trying to import numpy...\n";
        py::module_ np = py::module_::import("numpy");
        std::cout << "Numpy imported fine\n";
    } catch (const py::error_already_set &e) {
        std::cerr << e.what() << std::endl;
    }


    auto vec = generate5DClusters2DObvious(200, 10, 15, 0.5); //points, n clusters, dist between centers, noise

    auto pydata = create_2d_vector(vec);

    std::vector<std::vector<double>> umap_data;

    try {
        // Import your Python module (umap_test.py must be on PYTHONPATH)
        py::module umap_test = py::module::import("umap_test");

        // Call the run_umap function
        py::object result = umap_test.attr("run_umap")(
            pydata,    // inputdata
            "C:/Users/H/Desktop/cpp/pybind11 umap/plot2.png"      // output_plot
        );

        // Convert the returned Python dict to C++
        py::array_t<double> umap_result = result["reduced_data"].cast<py::array_t<double>>();

        // Access the data buffer and create a 2D vector
        auto r = umap_result.unchecked<2>(); // Unchecked access for fast iteration

        for(size_t i = 0; i < r.shape(0); ++i) {
           std::vector<double> row;
             for(size_t j = 0; j < r.shape(1); ++j) {
               row.push_back(r(i, j));
             }
                umap_data.push_back(row);
        } 

         std::cout << "UMAP result: " << umap_data.size() << " rows." << "Cols: " << umap_data[0].size() << std::endl;
     

    }
    catch (const py::error_already_set &e) {
        std::cerr << "Python error: " << e.what() << std::endl;
    }

    

}