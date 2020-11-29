#include <iostream>     // cout, endl
#include <vector>       // vector
#include <cstdlib>      // atoi

#include "loader.cpp"

using byte = std::byte;
template <typename T>
using vector = std::vector<T>;

int main(int argc, char** argv)
{
    int s2 = std::atoi(argv[1]);
    int trainingSize = 60000;
    
    const byte *const trainingLabels = LoadLabels("train-labels.idx1-ubyte");
    const byte *const trainingImages = LoadImages("train-images.idx3-ubyte");

    // @Note (jonathan): I'm getting decent performance here which is a bit surprising.
    //  My expectation is sizeof(float) * 10 * 60000 = 2,400,000 bytes
    vector<vector<float>> Y(trainingSize, vector<float>(10, 0.0f));
    auto labelIter = trainingLabels;
    for (int i = 0; i < trainingSize; ++i)
    {
        int label = (int)*labelIter;
        if (label == 0) label = 10;
        Y[i][label - 1] = 1.0f;
        ++labelIter;
    }
    
    vector<vector<byte>> inputLayer(28, vector<byte>(28, byte{0}));
    vector<vector<byte>> hiddenLayer(s2, vector<byte>(s2, byte{0}));
    vector<float> outputLayer(10, 0.0f);
    
    //byte *testLabels = LoadLabels("t10k-labels.idx1-ubyte");
    //byte *testImages = LoadImages("t10k-images.idx3-ubyte");
    
}