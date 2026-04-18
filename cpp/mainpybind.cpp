#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "engine.h"

namespace py = pybind11;

PYBIND11_MODULE(mga_module, m)
{
    py::class_<sysOutput>(m, "sysOutput")
        .def_readonly("regime", &sysOutput::regime)
        .def_readonly("description", &sysOutput::description)
        .def_readonly("action", &sysOutput::action)
        .def_readonly("G", &sysOutput::G)
        .def_readonly("E", &sysOutput::E)
        .def_readonly("C", &sysOutput::C)
        .def_readonly("variable_k1_weights", &sysOutput::variable_k1_weights)
        .def_readonly("variables", &sysOutput::variables)
        .def_readonly("dominant_eig_projection_coordinates_3dim", &sysOutput::dominant_eig_projection_coordinates_3dim)
        .def_readonly("eigen_axis1_weights", &sysOutput::eigen_axis1_weights)
        .def_readonly("eigen_axis2_weights", &sysOutput::eigen_axis2_weights)
        .def_readonly("eigen_axis3_weights", &sysOutput::eigen_axis3_weights)
        .def_readonly("eigen_axis1_contributors", &sysOutput::eigen_axis1_contributors)
        .def_readonly("eigen_axis2_contributors", &sysOutput::eigen_axis2_contributors)
        .def_readonly("variable_weights", &sysOutput::variable_weights)
        .def_readonly("eigen_axis3_contributors", &sysOutput::eigen_axis3_contributors)
        .def_readonly("structural_stability", &sysOutput::structural_stability)
        // api only
        .def_readonly("confidence", &sysOutput::confidence)
        .def_readonly("top_next_regimes", &sysOutput::top_next_regimes)
        .def_readonly("top_next_probs", &sysOutput::top_next_probs)
        .def_readonly("momentum_score", &sysOutput::momentum_score)
        .def_readonly("carry_score", &sysOutput::carry_score)
        .def_readonly("vol_score", &sysOutput::vol_score)
        .def_readonly("risk_posture", &sysOutput::risk_posture)
        .def_readonly("transition_risk", &sysOutput::transition_risk)
        .def_readonly("system_state_score", &sysOutput::system_state_score)
        .def_readonly("directional_bias", &sysOutput::directional_bias);


    py::class_<Engine>(m, "Engine")
        .def(py::init<>())
        .def("setHeaders", &Engine::setHeaders)
        .def("ingest", &Engine::ingest)
        .def("ready", &Engine::ready)
        .def("getOutput", &Engine::getOutput);
}