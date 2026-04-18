#include <vector>
#include <unordered_map>
#include <algorithm>
#include <string>
#include "stateanalysis.h"

//Temporal meta-structure on top of structural classification. Pattern evolution. Not predicion!

// helper: convert enum → index
inline int toIndex(CurRegime r)
{
    return static_cast<int>(r);
}

// helper: convert index → enum (optional safety layer)
inline CurRegime toRegime(int i)
{
    return static_cast<CurRegime>(i);
}

struct TransitionModel
{
    std::vector<std::vector<double>> P; // transition probabilities
    std::vector<std::pair<CurRegime, double>> top_next;
};

class MarkovRegimeModel
{
public:
    MarkovRegimeModel(int n, double decay = 0.995)
        : N(n), decay(decay)
    {
        counts.assign(N, std::vector<double>(N, 0.0));
        P.assign(N, std::vector<double>(N, 0.0));
    }

    void update(const std::vector<CurRegime> &history)
    {
        if (history.size() < 2)
            return;

        // decay old memory
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                counts[i][j] *= decay;

        // rebuild transitions from history
        for (size_t i = 1; i < history.size(); i++)
        {
            int from = toIndex(history[i - 1]);
            int to = toIndex(history[i]);

            if (valid(from) && valid(to))
                counts[from][to] += 1.0;
        }

        normalize();
    }

    TransitionModel predict(CurRegime current, int topK = 3)
    {
        TransitionModel out;

        int c = toIndex(current);
        if (!valid(c))
            return out;

        out.P = P;

        std::vector<std::pair<CurRegime, double>> probs;

        for (int j = 0; j < N; j++)
        {
            probs.push_back({toRegime(j), P[c][j]});
        }

        std::sort(probs.begin(), probs.end(),
                  [](auto &a, auto &b)
                  { return a.second > b.second; });

        for (int i = 0; i < std::min(topK, (int)probs.size()); i++)
            out.top_next.push_back(probs[i]);

        return out;
    }

private:
    int N;
    double decay;

    std::vector<std::vector<double>> counts;
    std::vector<std::vector<double>> P;

    bool valid(int i)
    {
        return i >= 0 && i < N;
    }

    void normalize()
    {
        const double threshold = 5.0; // minimum transitions required

        for (int i = 0; i < N; i++)
        {
            double sum = 0.0;

            for (int j = 0; j < N; j++)
                sum += counts[i][j];

            if (sum < threshold)
            {
                // not enough data → zero out row
                for (int j = 0; j < N; j++)
                    P[i][j] = 0.0;

                continue;
            }

            for (int j = 0; j < N; j++)
                P[i][j] = counts[i][j] / sum;
        }
    }
};

/*
how i should i use it

MarkovRegimeModel model(15); // number of regimes

model.update(history);

auto result = model.predict(current_regime);

for (auto& [regime, prob] : result.top_next)
{
    std::cout << (int)regime << " -> " << prob << "\n";
}

StableTrend →
    StableTrend       0.62
    Exhaustion        0.21
    Transitioning     0.10

*/