#include <iostream>     // cout, endl
#include <vector>       // vector
#include <cstdlib>      // atoi
#include <cmath>        // exp

#include "loader.cpp"
#include "prng.cpp"

using byte = std::byte;
template <typename T>
using vector = std::vector<T>;

vector<vector<byte>> operator*(vector<vector<byte>> lhs, vector<vector<byte>> rhs)
{
    return lhs;
}

vector<float> Sigmoid(const vector<float> &vec)
{
    vector<float> output = vec;
    for (auto& i : output) i = 1 / (1 + std::exp(-i));
    return output;
}

int main(int argc, char** argv)
{
    int s2 = std::atoi(argv[1]);
    int trainingSize = 60000;
    int imageSize = 784;
    
    const byte *const trainingLabels = LoadLabels("train-labels.idx1-ubyte");
    const byte *const trainingImages = LoadImages("train-images.idx3-ubyte");
    
    vector<vector<float>> Y(trainingSize, vector<float>(10, 0.0f));
    {
        auto labelIter = trainingLabels;
        for (int i = 0; i < trainingSize; ++i)
        {
            int label = (int)*labelIter;
            if (label == 0) label = 10;
            Y[i][label - 1] = 1.0f;
            ++labelIter;
        }
    }
    
    vector<vector<float>> inputLayer(trainingSize + 1, vector<float>(imageSize, 1.0f));
    {
        auto imageIter = trainingImages;
        for (int image = 0; image < trainingSize; ++image)
        {
            for (int pixel; pixel < imageSize; ++pixel)
            {
                inputLayer[image + 1][pixel] = (float)*imageIter / 255.0f;
            }
        }
    }
    
    vector<vector<float>> W1(trainingSize, vector<float>(trainingSize + 1));
    {
        for (auto& i : W1)
        {
            for (auto& j : i)
            {
                j = prng();
            }
        }
    }
    
    vector<vector<float>> hiddenLayer(trainingSize + 1, vector<float>(s2));
    
    vector<float> outputLayer(10, 0.0f);
    
    //byte *testLabels = LoadLabels("t10k-labels.idx1-ubyte");
    //byte *testImages = LoadImages("t10k-images.idx3-ubyte");
    
    
}