#include <random>   // mt19937, uniform_real_distribution, random_device
#include <chrono>   // 

// @Source: https://stackoverflow.com/questions/13445688/how-to-generate-a-random-number-in-c
float prng()
{
    using namespace std::chrono;
    
    std::random_device rd;
    auto s = duration_cast<seconds>
        (system_clock::now().time_since_epoch()).count();
    auto ms = duration_cast<microseconds>
        (high_resolution_clock::now().time_since_epoch()).count();

    std::mt19937::result_type seed = rd() ^ (s + ms);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(0.0001f, 0.01f);
    return dis(gen);
}
