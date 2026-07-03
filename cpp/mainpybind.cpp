#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "engine.h"

namespace py = pybind11;

// Thin view structs — zero-copy lenses over sysOutput for nested Python access.
// The C++ struct stays flat. These exist only in the binding layer.
struct DecisionView  { const sysOutput* s; };
struct RegimeView    { const sysOutput* s; };
struct StressView    { const sysOutput* s; };
struct FactorsView   { const sysOutput* s; };
struct StructureView { const sysOutput* s; };

PYBIND11_MODULE(mga_module, m)
{
    // --- ClientConfig ---
    py::class_<ClientConfig>(m, "ClientConfig")
        .def(py::init<>())
        // TIME_HORIZON
        .def_readwrite("window_size",              &ClientConfig::window_size)
        .def_readwrite("window_stride",            &ClientConfig::window_stride)
        .def_readwrite("robust_history",           &ClientConfig::robust_history)
        .def_readwrite("robust_warmup",            &ClientConfig::robust_warmup)
        .def_readwrite("lle_history_depth",        &ClientConfig::lle_history_depth)
        .def_readwrite("regime_confidence_depth",  &ClientConfig::regime_confidence_depth)
        .def_readwrite("stability_window",         &ClientConfig::stability_window)
        .def_readwrite("swirl_window",             &ClientConfig::swirl_window)
        // SENSITIVITY
        .def_readwrite("regime_confirm_count",       &ClientConfig::regime_confirm_count)
        .def_readwrite("regime_decay_step",          &ClientConfig::regime_decay_step)
        .def_readwrite("gec_ema_alpha",              &ClientConfig::gec_ema_alpha)
        .def_readwrite("normalizer_base_alpha",      &ClientConfig::normalizer_base_alpha)
        .def_readwrite("normalizer_min_alpha",       &ClientConfig::normalizer_min_alpha)
        .def_readwrite("normalizer_decay",           &ClientConfig::normalizer_decay)
        .def_readwrite("normalizer_update_interval", &ClientConfig::normalizer_update_interval)
        // THRESHOLD
        .def_readwrite("pca_variance_retained",  &ClientConfig::pca_variance_retained)
        .def_readwrite("eigengap_min_ratio",     &ClientConfig::eigengap_min_ratio)
        .def_readwrite("lle_shift_cap",          &ClientConfig::lle_shift_cap)
        .def_readwrite("stability_decay_k",      &ClientConfig::stability_decay_k)
        .def_readwrite("min_duration_episodes",  &ClientConfig::min_duration_episodes)
        .def_readwrite("swirl_speed_floor",      &ClientConfig::swirl_speed_floor)
        .def_readwrite("pc1_significance",       &ClientConfig::pc1_significance)
        .def_readwrite("lle_r2_threshold",       &ClientConfig::lle_r2_threshold)
        // WEIGHT — exposed as Python lists via pybind11/stl.h array binding
        .def_readwrite("g_weights", &ClientConfig::g_weights)
        .def_readwrite("e_weights", &ClientConfig::e_weights)
        .def_readwrite("t_weights", &ClientConfig::t_weights);

    // --- EngineProfile enum ---
    py::enum_<EngineProfile>(m, "EngineProfile")
        .value("LONG_SHORT_EQUITY", EngineProfile::LONG_SHORT_EQUITY)
        .value("GLOBAL_MACRO",      EngineProfile::GLOBAL_MACRO)
        .value("SYSTEMATIC_TREND",  EngineProfile::SYSTEMATIC_TREND)
        .value("RISK_PARITY",       EngineProfile::RISK_PARITY);

    // --- profile_defaults() ---
    m.def("profile_defaults", &profile_defaults, py::arg("profile"),
          "Returns a fully populated ClientConfig with the profile's defaults.");

    // --- ConfigViolation ---
    py::class_<ConfigViolation>(m, "ConfigViolation")
        .def_readonly("field",   &ConfigViolation::field)
        .def_readonly("message", &ConfigViolation::message)
        .def_readonly("value",   &ConfigViolation::value)
        .def_readonly("bound",   &ConfigViolation::bound)
        .def("__repr__", [](const ConfigViolation& v) {
            return "<ConfigViolation field='" + v.field + "' message='" + v.message + "'>";
        });

    // --- validate() ---
    m.def("validate", &validate, py::arg("config"),
          "Returns all violations. Empty list = valid config.");

    // --- state.decision (5 fields) — what a PM reads at 8:30am ---
    py::class_<DecisionView>(m, "_DecisionView")
        .def_property_readonly("action",           [](const DecisionView& v) { return v.s->decision_action; })
        .def_property_readonly("rationale",        [](const DecisionView& v) { return v.s->decision_regime_rationale; });

    // --- state.regime (4 fields) — current regime state ---
    py::class_<RegimeView>(m, "_RegimeView")
        .def_property_readonly("current",    [](const RegimeView& v) { return v.s->decision_regime; })
        .def_property_readonly("confidence", [](const RegimeView& v) { return v.s->regime_confidence; })
        .def_property_readonly("age",        [](const RegimeView& v) { return v.s->regime_age; });


    // --- state.structure (11 fields) — eigenvector decomposition + projection ---
    py::class_<StructureView>(m, "_StructureView")
        .def_property_readonly("input_variables",         [](const StructureView& v) { return v.s->model_input_variables; })
        .def_property_readonly("raw_input",               [](const StructureView& v) { return v.s->model_raw_input; })
        .def_property_readonly("pc1_weights_smoothed",    [](const StructureView& v) { return v.s->model_pc1_weights_smoothed; });

    // --- Top-level sysOutput: nested views via reference_internal ---
    py::class_<sysOutput>(m, "sysOutput")
        .def_property_readonly("decision",  [](const sysOutput& s) { return DecisionView{&s};  }, py::return_value_policy::reference_internal)
        .def_property_readonly("regime",    [](const sysOutput& s) { return RegimeView{&s};    }, py::return_value_policy::reference_internal)
        .def_property_readonly("stress",    [](const sysOutput& s) { return StressView{&s};    }, py::return_value_policy::reference_internal)
        .def_property_readonly("factors",   [](const sysOutput& s) { return FactorsView{&s};   }, py::return_value_policy::reference_internal)
        .def_property_readonly("structure", [](const sysOutput& s) { return StructureView{&s}; }, py::return_value_policy::reference_internal);

    // --- Engine ---
    py::class_<Engine>(m, "Engine")
        .def(py::init<const ClientConfig&>(), py::arg("config"))
        .def("setHeaders", &Engine::setHeaders)
        .def("ingest", &Engine::ingest)
        .def("ready", &Engine::ready)
        .def("get_snapshot", &Engine::getOutput);
}