#include "utils.h"
#include "engine.h"

int main() //purely for testing real quick with c++ speed
{

    Engine engine;

    std::ifstream file("C:/Users/H/Documents/cpp/v_output/etf_daily_log_returns.csv"); // C:/Users/H/Documents/cpp/v_output/etf_daily_log_returns_stable_period.csv or etf_daily_log_returns.csv or etf_daily_log_returns_shuffled.csv
    if (!file.is_open())
    {
        std::cerr << "Could not open file\n";
        throw std::exception("couldnt open file");
    }

    std::string line;

    std::getline(file, line); // read header (and skip it)
    std::stringstream ss_h(line);
    std::string token_h;
    std::vector<std::string> headers;
    int counter = 0;

    while (std::getline(ss_h, token_h, ','))
    {
        if (counter++ == 0) // check, and then increment
            continue;

        std::cerr << token_h << " ";
        headers.push_back(token_h);
    }

    LOG("HEADERS: ");

    for (const auto &header : headers)
    {
        std::cerr << header << std::endl;
    }

    engine.setHeaders(headers);

    while (std::getline(file, line))
    {
        if (line.empty())
            continue;

        std::stringstream ss(line);
        std::string token;
        std::vector<double> row;
        bool firstColumn = true;

        while (std::getline(ss, token, ','))
        {
            if (firstColumn)
            {
                firstColumn = false; // skip time column
                continue;
            }

            if (token.empty())
                continue;

            try
            {
                row.push_back(std::stod(token));
            }
            catch (...)
            {
                // bad token, skip it
            }
        }

        engine.ingest(row);

        if (engine.ready())
        {
            auto out = engine.getOutput();
        }
    }

    return 0;
}