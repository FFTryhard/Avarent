#include "engine.h"
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
#include "utils.h"
#include <stdexcept>
#include "rawprocessing2.h"

// All tuning parameters now come from ClientConfig (stored as this->config).
// See client_config.h for definitions, bounds, and profile defaults.

// See client_config.h for definitions, bounds, and profile defaults.

bool Engine::ready() { return output_ready; } // can we output?

void Engine::setHeaders(std::vector<std::string> x)
{
  headers = x;
  headers_are_set = true;
}

std::vector<std::string> Engine::getHeaders()
{
  if (headers_are_set)
  {
    return headers;
  }
  throw std::runtime_error("HEADERS ARE NOT SET CANT RETURN\n");
  return {};
}

sysOutput Engine::getOutput()
{
  output_ready = false; // optional: reset after read
  return current_output;
} // current result

void Engine::ingest(const std::vector<double> &row)
{
  if (!row.empty())
  {

    std::vector<double> raw_vec = in.process(row);

    if (checkForNaN(raw_vec))
    {
      std::cerr << "Raw Vec Empty/Full of Zero's or NaN's!" << std::endl;
      for (int i = 0; i < raw_vec.size(); i++)
      {
        std::cerr << "R[" << i << "]: " << raw_vec[i] << "\n";
      }
      std::cerr << "Index: " << iteration << std::endl;
    }

    // new
    NormalizerState state = in2.process(row);

    // --- ONLINE G AND C (with k-spacing macro deltas) ---
    state_history.push_back(state);
    const int k_spacing = config.window_stride; // use stride (e.g. 20) as the macro window
    if ((int)state_history.size() > k_spacing)
    {
      state_history.pop_front();
    }

    double G2 = 0.0;
    double C2 = 0.0;

    if ((int)state_history.size() == k_spacing)
    {
      const NormalizerState &past_state = state_history.front();
      if (past_state.ready && state.ready)
      {
        G2 = computeG(past_state.eigenvectors, state.eigenvectors, state.eigenvalues);
        C2 = computeC(past_state.eigenvalues, state.eigenvalues);
      }
    }

    auto z_g2 = g2_score.process(G2, RobustScore::Mode::Positive);
    auto z_c2 = c2_score.process(C2, RobustScore::Mode::Signed); // Was signed before

    double g2_norm = z_g2.valid ? z_g2.z : 0.0;
    double c2_norm = z_c2.valid ? z_c2.z : 0.0;

    g2_ema.update(g2_norm);
    if (g2_ema.ready())
      g2_norm = g2_ema.get();

    c2_ema.update(c2_norm);
    if (c2_ema.ready())
      c2_norm = c2_ema.get();
    // ----------------------

    /*

    double G2 = computeG(in2.prevEigenvectors(), state.eigenvectors, state.eigenvalues);
    double C2 = computeC(in2.prevEigenvalues(), state.eigenvalues);

    g2_ema.update(G2);
    if (g2_ema.ready()) G2 = g2_ema.get();

    c2_ema.update(C2);
    if (c2_ema.ready()) C2 = c2_ema.get();

    auto z_g2 = g2_score.process(G2, RobustScore::Mode::Positive);
    auto z_c2 = c2_score.process(C2, RobustScore::Mode::Absolute);

    double g2_norm = z_g2.valid ? z_g2.z : 0.0;
    double c2_norm = z_c2.valid ? z_c2.z : 0.0;

    */

    WINDOW.push_back(raw_vec); // push until full. was raw_vec

    if ((int)WINDOW.size() >= config.window_size) // when full, analyze that block
    {
      std::cerr << "Iteration: " << iteration << '\n';

      // DATA FOR LLE
      Eigen::MatrixXd final_LLE_data_matrix;

      auto data_std_no_pca = deque_to_vector(WINDOW);
      auto data_eigen_no_pca = dequeToEigenMatrix(WINDOW);

      final_LLE_data_matrix = data_eigen_no_pca;
      if (getHeaders().size() >= 30)
      {
        auto pca_result = PCA_Full(data_eigen_no_pca, config.pca_variance_retained);
        auto data_std = std::move(pca_result.projected);
        auto data_eigen = vectorToEigenMatrix(data_std);

        final_LLE_data_matrix = data_eigen;
      }

      if (containsNaN(data_std_no_pca) || containsINF(data_std_no_pca) || WINDOW.empty())
      {
        LOG("Window has NaN/Inf values, bailing analysis.");
        return;
      }

      // PATTERN ANLYSIS
      int raw_start = iteration - config.window_size + 1;
      int raw_center = raw_start + config.window_size / 2;

      auto cov = covarianceMatrixx(data_std_no_pca);

      double rolvol = mean(volatilityFromCov(cov));
      double eigratio = eigenvalueDominance(data_eigen_no_pca);
      double MAR = mean(eigenToStdVector(meanAbsoluteReturn(data_eigen_no_pca)));

      //--------

      WINDOW.pop_front(); // Stride = 1 (rolling daily window)

      std::cerr << std::endl;
      std::cerr << std::endl;

      double E2 = computeE(state.maha);

      auto LLE = largest_lyapunov_rosenstein(final_LLE_data_matrix, 1);
      double E = LLE.lle;

      auto z_e2 = e2_score.process(E2, RobustScore::Mode::Positive);
      auto z_e = e_score.process(E, RobustScore::Mode::Positive);

      double e2_norm = z_e2.valid ? z_e2.z : 0.0;
      double e_norm = z_e.valid ? z_e.z : 0.0;

      e2_ema.update(e2_norm);
      if (e2_ema.ready())
        e2_norm = e2_ema.get();

      e_ema.update(e_norm);
      if (e_ema.ready())
        e_norm = e_ema.get();

      /*
      e2_ema.update(E2);
      if (e2_ema.ready()) E2 = e2_ema.get();

      e_ema.update(E);
      if (e_ema.ready()) E = e_ema.get();

      auto z_e2 = e2_score.process(E2, RobustScore::Mode::Positive);
      auto z_e =  e_score.process(E, RobustScore::Mode::Positive);

      double e2_norm = z_e2.valid ? z_e2.z : 0.0;
      double e_norm = z_e.valid ? z_e.z : 0.0;
      */

      // std::cout << raw_center << " " << g2_norm + 2.5 << " " << raw_center << " " << e_norm << " " << raw_center << " " << c2_norm << " " << std::endl;

      htts.push_back({g2_norm, e_norm, c2_norm}); // if i smoothed it out too much, this should reveal clusters better

      if (z_g2.valid && z_c2.valid && z_e.valid)
      {
        gec_window.push_back({g2_norm, e_norm, c2_norm});
        if (gec_window.size() >= (size_t)config.hmmwindow)
        {
          try
          {
            py::gil_scoped_acquire gil;

            std::cout << previous_means.rows()
                      << " x "
                      << previous_means.cols()
                      << std::endl;

            Eigen::MatrixXd matched = fit_and_match(gec_window, previous_means).cast<Eigen::MatrixXd>();

            LOG(matched);

            previous_means = matched;

            gec_window.erase(gec_window.begin(), gec_window.begin() + config.hmmwindowstride);
          }
          catch (const py::error_already_set &e)
          {
            std::cerr << e.what() << std::endl;
            throw;
          }
        }

        auto detected_regime = classifyRegime(g2_norm, e_norm, c2_norm, reversetransform(previous_means));
        if (!state.ready)
          detected_regime = CurRegime::Neutral;
        auto current_regime_stable = updateRegime(detected_regime);

        // Regime age tracking
        if (!regime_tracking_init)
        {
          last_tracked_regime = current_regime_stable;
          regime_tracking_init = true;
          regime_age_counter = 0;
        }
        else if (current_regime_stable != last_tracked_regime)
        {
          regime_durations[static_cast<int>(last_tracked_regime)].push_back(
              regime_age_counter);
          regime_age_counter = 0;
          last_tracked_regime = current_regime_stable;
        }
        else
        {
          regime_age_counter++;
        }

        previous_regime = current_regime_stable;

        // --- PC1 signal classification ---
        // 3.1: Smooth the PC1 weights using EMA (reuses GEC_EMA_ALPHA)
        Eigen::VectorXd pc1_raw = state.eigenvectors.col(state.eigenvalues.size() - 1);
        if (!pc1_smoothed_init)
        {
          pc1_smoothed = pc1_raw;
          pc1_smoothed_init = true;
        }
        else
        {
          // Eigenvectors have arbitrary signs. Align sign with the previous state before EMA
          if (pc1_raw.dot(pc1_smoothed) < 0.0)
          {
            pc1_raw = -pc1_raw;
          }
          pc1_smoothed = config.gec_ema_alpha * pc1_raw +
                         (1.0 - config.gec_ema_alpha) * pc1_smoothed;
        }

        // 3.2: Classify instruments by significance
        double mean_abs_w = pc1_smoothed.cwiseAbs().mean();
        double sig_threshold = config.pc1_significance * mean_abs_w;

        struct PC1Instrument
        {
          int index;
          double weight;
        };

        std::vector<PC1Instrument> reinforcing, antagonistic;
        for (int i = 0; i < pc1_smoothed.size() && i < (int)headers.size();
             ++i)
        {
          double w = pc1_smoothed(i);
          if (std::abs(w) >= sig_threshold)
          {
            if (w > 0.0)
              reinforcing.push_back({i, w});
            else
              antagonistic.push_back({i, w});
          }
        }

        // 3.3: Select top 2 from each group by absolute weight
        auto by_abs_weight = [](const PC1Instrument &a,
                                const PC1Instrument &b)
        {
          return std::abs(a.weight) > std::abs(b.weight);
        };
        std::sort(reinforcing.begin(), reinforcing.end(), by_abs_weight);
        std::sort(antagonistic.begin(), antagonistic.end(), by_abs_weight);

        if (reinforcing.size() > 2)
          reinforcing.resize(2);
        if (antagonistic.size() > 2)
          antagonistic.resize(2);

        // 3.4: Generate description string
        std::string rationale;
        bool has_r = !reinforcing.empty();
        bool has_a = !antagonistic.empty();

        if (!has_r && !has_a)
        {
          rationale = "Market structure flat. No dominant dynamic detected.";
        }
        else if (has_r && has_a)
        {
          rationale = "Regime driven by " + headers[reinforcing[0].index];
          if (reinforcing.size() > 1)
            rationale += ", " + headers[reinforcing[1].index];
          rationale += ". " + headers[antagonistic[0].index];
          if (antagonistic.size() > 1)
            rationale += ", " + headers[antagonistic[1].index];
          rationale += " acting against dominant dynamic.";
        }
        else if (has_r && !has_a)
        {
          rationale = "Regime driven by " + headers[reinforcing[0].index];
          if (reinforcing.size() > 1)
            rationale += ", " + headers[reinforcing[1].index];
          rationale += ". No significant antagonistic pressure.";
        }
        else
        { // !has_r && has_a
          rationale =
              "Dominant dynamic unclear. " + headers[antagonistic[0].index];
          if (antagonistic.size() > 1)
            rationale += ", " + headers[antagonistic[1].index];
          rationale += " acting against prevailing structure.";
        }

        current_output.decision_regime = enum_to_string_regime(current_regime_stable);
        current_output.decision_action = interpret(current_regime_stable); // maybe add explanation later
        current_output.model_input_variables = getHeaders();
        current_output.model_raw_input = raw_vec;
        // current_output.regime_confidence = r_confidence; //based on coordinate location cossim and or distance? comes later
        // current_output.decision_regime_rationale = re-anta;
        current_output.regime_age = regime_age_counter;
        current_output.model_pc1_weights_smoothed = eigenToStdVector(pc1_smoothed);

        output_ready = true;
      }

      std::cerr << std::endl;
      std::cerr << std::endl;
    }
  }
  iteration++;
}

CurRegime Engine::updateRegime(CurRegime detected_regime)
{
  // Initialization
  if (!regime_init)
  {
    confirmed_regime = detected_regime;
    regime_candidate = detected_regime;
    regime_score = config.regime_confirm_count; // fully confident at start
    regime_init = true;
    return confirmed_regime;
  }

  // If matches current confirmed → reinforce stability
  if (detected_regime == confirmed_regime)
  {
    regime_candidate = detected_regime;
    regime_score = std::min(regime_score + 1, config.regime_confirm_count);
  }
  else
  {
    // If matches candidate → build confidence
    if (detected_regime == regime_candidate)
    {
      regime_score = std::min(regime_score + 1, config.regime_confirm_count);
    }
    else
    {
      // New challenger → decay instead of reset
      regime_candidate = detected_regime;
      regime_score =
          std::max(regime_score - config.regime_decay_step, SCORE_MIN);
    }

    // Switch only when confidence is high enough
    if (regime_score >= config.regime_confirm_count)
    {
      confirmed_regime = regime_candidate;
      regime_score = config.regime_confirm_count / 2; // avoid instant flip back
    }
  }

  return confirmed_regime;
}

