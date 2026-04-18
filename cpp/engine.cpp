#include "utils.h"
#include "engine.h"

bool Engine::ready()
{
    return output_ready;
} // can we output?

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
    std::cerr << "HEADERS ARE NOT SET CANT RETURN\n";
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

        WINDOW.push_back(raw_vec); // push until full

        if (WINDOW.size() >= EFFECTIVE_BUFFER) // when full, analyze that block
        {
            std::cerr << "Iteration: " << iteration << '\n';

            auto data_std_no_pca = deque_to_vector(WINDOW);
            auto data_eigen_no_pca = dequeToEigenMatrix(WINDOW);

            auto data_std = PCA(data_eigen_no_pca, 0.90);
            auto data_eigen = vectorToEigenMatrix(data_std);

            if (containsNaN(data_std_no_pca) || containsINF(data_std_no_pca))
            {
                LOG("Window has NaN/Inf values, bailing analysis.");
                return;
            }

            auto LLE = largest_lyapunov_rosenstein(data_eigen, 1);

            bool lle_shift_valid = false;
            double z_lle = 0.0;

            auto [boolean, r2] = stabilityCheckLLE(LLE);

            if (boolean)
            {
                lle_regime.push_back(LLE.lle);

                if (lle_regime.size() >= N)
                {
                    z_lle = splitMedianMadChange(deque_to_1vector(lle_regime)); // is it contracting or expanding or neutral. this will answer it
                    lle_shift_valid = true;
                    for (int i = 0; i < lle_regime_stride; i++)
                    {
                        lle_regime.pop_front();
                    }
                }
            }
            else
            {
                LOG("Window too unstable for accurate LLE estimation.");
            }

            auto trajectory = computeSwirlSeverityWindowed(data_eigen_no_pca, 1, 200, 50, 0.10); // was 200

            auto [U, lambdas] = PCA_Eigenvectors(data_eigen_no_pca, 0.90); // needs to be pca on this window

            double gap = lambdas[0] - lambdas[1];

            double sum_eig = 0.0;

            for (const auto &x : lambdas)
            {
                sum_eig = sum_eig + x;
            }

            if (U.rows() == 0 || U.cols() == 0)
            {
                std::cerr << "Skipping: empty U\n";
                return; // default this
            }

            U_buff.add(U);
            Uvec_buff.add(U.col(0));

            double sin_normDS = 0.0;  // wedin
            double sin_normMOS = 0.0; // wedin

            bool dominance_ok_right_now = true;

            if (gap / (sum_eig + 1e-11) < 1e-3) // above is fine
            {                                   // should ass this in dd too
                dominance_ok_right_now = false;
            }

            if (!(U_buff.has_prev) || !dominance_ok_right_now) // add dot product
            {
                // Not enough history yet
                LOG("DS/MOS not stable due to eigenvalue gap");
                sin_normDS = 0.0;
                sin_normMOS = 0.0;
            }
            else
            {
                Eigen::MatrixXd U_overlap = U_buff.previous().transpose() * U_buff.current(); // UtU

                sin_normMOS = wedin_theoremKGT1(U_overlap);                                                                // this never gets activated
                sin_normDS = wedin_theoremK1(Eigen::VectorXd(Uvec_buff.current()), Eigen::VectorXd(Uvec_buff.previous())); // just needs dot product of vectors
                // DSsinnorms.push_back(wedin_marginDS); // MOSsinnorms.push_back(wedin_marginMOS); histograms
            }

            double directional_dominance = std::log((lambdas[0] / (lambdas[1] + 1e-12)) + 1e-12);
            int d = (int)lambdas.size();

            // G(t)
            double ds = sin_normDS;
            double mos = sin_normMOS; // used for transitions, and periodic executions. 2 .. -2 then just transitions
            double dd = directional_dominance;

            // deltas
            ddelta_buff.add(dd);
            double ddelta = ddelta_buff.hasPrevious() ? (dd - ddelta_buff.previous()) : 0.0;

            // E(t)
            double srs = trajectory.empty() ? 0.0 : median(trajectory);
            double lle = boolean ? LLE.lle : std::numeric_limits<double>::quiet_NaN(); // r2 there too
            double lle_shift = z_lle;                                                  // handled above

            // C(t)
            double d_eff = effective_Rank(lambdas);

            // deltas
            dde_delta_buff.add(d_eff); // used for transitions and when to execute
            double dde_delta = dde_delta_buff.hasPrevious() ? (d_eff - dde_delta_buff.previous()) : 0.0;

            // NORMALZING
            auto z_ds = ds_score.process(ds, RobustScore::Mode::Positive);
            auto z_mos = mos_score.process(mos, RobustScore::Mode::Positive);
            auto z_dd = ddelta_score.process(ddelta, RobustScore::Mode::Absolute);
            auto z_de = deff_delta_score.process(dde_delta, RobustScore::Mode::Absolute);
            auto z_srs = srs_score.process(srs, RobustScore::Mode::Positive);
            auto z_lle_level = 0.0;
            if (boolean && std::isfinite(lle))
            {
                auto out = lle_level_score.process(lle, RobustScore::Mode::Positive);
                if (out.valid)
                    z_lle_level = out.z;
            }
            auto z_lle_shift = 0.0;
            if (boolean && lle_shift_valid)
            {
                z_lle_shift = std::max(0.0, std::min(8.0, z_lle));
            }

            // SCORING
            double G = 0.0;
            if (z_ds.valid && z_mos.valid && z_dd.valid)
            {
                G = 0.4 * z_ds.z + 0.4 * z_mos.z + 0.2 * z_dd.z;
            }
            g.update(G);
            if (g.ready())
                G = g.get();

            double E = 0.0;
            double s_lle = z_lle_level;
            double s_lle_shift = z_lle_shift;
            double s_srs = z_srs.valid ? z_srs.z : 0.0;

            E = 0.35 * s_lle + 0.40 * s_lle_shift + 0.25 * s_srs;
            e.update(E);
            if (e.ready())
                E = e.get();

            double C = 0.0;
            if (z_de.valid)
            {
                C = z_de.z;
            }
            c.update(C);
            if (c.ready())
                C = c.get();

            G_bin.push(G);
            E_bin.push(E);
            C_bin.push(C);

            int CURRENT_STATE_G = G_bin.exec();
            int CURRENT_STATE_E = E_bin.exec();
            int CURRENT_STATE_C = C_bin.exec();

            gec_states.setG(CURRENT_STATE_G);
            gec_states.setE(CURRENT_STATE_E);
            gec_states.setC(CURRENT_STATE_C);

            gec_states.update();

            // PATTERN ANLYSIS
            int raw_start = window_index * BUFFER_STRIDE;
            int raw_center = raw_start + EFFECTIVE_BUFFER / 2;

            int HOW_MANY_PATTERNS_DOMINATE = std::round(d_eff);

            Patterns p;
            p.eff = HOW_MANY_PATTERNS_DOMINATE;
            p.eigenValues = lambdas;
            p.eigenVectors = U;
            p.headers = headers;

            double T_raw = 0.35 * G + 0.40 * E + 0.25 * C;

            t.update(T_raw);
            double T_raw_smoothed = 0.0; // probably dont need T
            if (t.ready())
                T_raw_smoothed = t.get();

            // dumb baselines

            auto cov = covarianceMatrixx(data_std_no_pca);
            double rolvol = mean(volatilityFromCov(cov));

            double eigratio = eigenvalueDominance(data_eigen_no_pca);

            double MAR = mean(eigenToStdVector(meanAbsoluteReturn(data_eigen_no_pca)));

            //--------

            std::cout << raw_center << " " << G << " " << raw_center << " " << E << " " << raw_center << " " << C << " " << raw_center << " " << T_raw_smoothed << " " << raw_center << " " << MAR << std::endl;

            std::vector<double> coords = {G, E, C};
            std::vector<double> coordsB = {eigratio, MAR, rolvol};

            for (size_t i = 0; i < BUFFER_STRIDE; ++i)
                WINDOW.pop_front();

            std::cerr << "Window index: " << window_index << std::endl;
            window_index++;

            std::cerr << std::endl;
            std::cerr << std::endl;

            if (gec_states.isReady())
            {
                auto detected_regime = classifyRegime(gec_states);
                auto current_regime_stable = updateRegime(detected_regime);

                regimehistory.push_back(current_regime_stable);
                double r_confidence = -1; // init

                if (regimehistory.size() == regimehistory_N)
                {
                    r_confidence = temporalConsistency(regimehistory);
                    for (size_t i = 0; i < regimehistory_stride; ++i)
                        regimehistory.pop_front();
                }

                markovregimehistory.push_back(current_regime_stable);
                std::vector<std::string> top_r = {"Initializing", "Initializing", "Initializing"};
                std::vector<double> top_prob = {-1, -1, -1}; // set to three anyway

                double transition_risk = 0.5;

                if (markovregimehistory.size() == markovregimehistory_N)
                {
                    top_r.clear();
                    top_prob.clear();

                    markovmodel.update(deque_to_vector_template(markovregimehistory));
                    auto result = markovmodel.predict(current_regime_stable); // markov property. From THIS regime, the three next are most likely now

                    int c = toIndex(current_regime_stable);

                    if (c >= 0 && c < result.P.size())
                    {
                        double row_sum = 0.0;
                        for (double v : result.P[c])
                            row_sum += v;

                        if (row_sum > 0.0)
                        {
                            double stay_prob = result.P[c][c];
                            transition_risk = 1.0 - stay_prob;
                        }
                    }

                    for (auto &[regime, prob] : result.top_next)
                    {
                        top_r.push_back(enum_to_string_regime(regime));
                        top_prob.push_back(prob);
                    }

                    for (size_t i = 0; i < markovregimehistory_stride; ++i)
                        markovregimehistory.pop_front();
                }

                current_output.regime = enum_to_string_regime(current_regime_stable);
                current_output.action = action(current_regime_stable);
                current_output.description = interpret(current_regime_stable);
                current_output.G = G;
                current_output.E = E;
                current_output.C = C;
                current_output.variable_k1_weights = eigenToStdVector(p.eigenVectors.col(0));
                current_output.variables = getHeaders();

                auto [e1, e2, e3] = getTop3EigenVectors(p);

                auto e1_topweights = getTopK(e1, tracker_axis1, 3);
                auto e2_topweights = getTopK(e2, tracker_axis2, 3);
                auto e3_topweights = getTopK(e3, tracker_axis3, 3);

                auto [eigen_axis1_contributors, eigen_axis1_weights] = getAxisInfo(e1_topweights, headers); // x
                auto [eigen_axis2_contributors, eigen_axis2_weights] = getAxisInfo(e2_topweights, headers); // y
                auto [eigen_axis3_contributors, eigen_axis3_weights] = getAxisInfo(e3_topweights, headers); // z

                current_output.eigen_axis1_contributors = eigen_axis1_contributors;
                current_output.eigen_axis2_contributors = eigen_axis2_contributors;
                current_output.eigen_axis3_contributors = eigen_axis3_contributors;
                current_output.eigen_axis1_weights = eigen_axis1_weights;
                current_output.eigen_axis2_weights = eigen_axis2_weights;
                current_output.eigen_axis3_weights = eigen_axis3_weights;

                current_output.dominant_eig_projection_coordinates_3dim = eigenStateProjections3dim(e1, e2, e3, toEigenVector(raw_vec)); // x y z

                current_output.variable_weights = raw_vec;

                current_output.confidence = r_confidence;

                current_output.top_next_regimes = top_r;
                current_output.top_next_probs = top_prob; // in order!;

                t_raw_smoothed.push_back(T_raw_smoothed);
                double regime_stability = -1; // init

                if (t_raw_smoothed.size() == t_raw_smoothed_N)
                {
                    double instability = std::abs(derivativeOLS(deque_to_1vector(t_raw_smoothed), 1.0)); // dt

                    double k = 5.0; // scale factor, tune
                    regime_stability = std::exp(-k * instability);
                    regime_stability = std::max(0.0, std::min(1.0, regime_stability));

                    for (size_t i = 0; i < t_raw_smoothed_stride; ++i)
                        t_raw_smoothed.pop_front();
                }

                current_output.structural_stability = regime_stability;

                current_output.momentum_score = clamp_((1.0 - G) * regime_stability * (1.0 - C));
                current_output.carry_score = clamp_(regime_stability * (1.0 - C));
                current_output.vol_score = clamp_(C);

                current_output.directional_bias = cosineSimilarity(toEigenVector(raw_vec), p.eigenVectors.col(0));

                current_output.risk_posture = 1 - transition_risk;

                current_output.transition_risk = transition_risk;

                current_output.system_state_score = r_confidence * regime_stability* (1 - transition_risk) * (1 - C);

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
        regime_score = SCORE_MAX; // fully confident at start
        regime_init = true;
        return confirmed_regime;
    }

    // If matches current confirmed → reinforce stability
    if (detected_regime == confirmed_regime)
    {
        regime_candidate = detected_regime;
        regime_score = std::min(regime_score + 1, SCORE_MAX);
    }
    else
    {
        // If matches candidate → build confidence
        if (detected_regime == regime_candidate)
        {
            regime_score = std::min(regime_score + 1, SCORE_MAX);
        }
        else
        {
            // New challenger → decay instead of reset
            regime_candidate = detected_regime;
            regime_score = std::max(regime_score - DECAY_STEP, SCORE_MIN);
        }

        // Switch only when confidence is high enough
        if (regime_score >= SCORE_MAX)
        {
            confirmed_regime = regime_candidate;
            regime_score = SCORE_MAX / 2; // avoid instant flip back
        }
    }

    return confirmed_regime;
}
