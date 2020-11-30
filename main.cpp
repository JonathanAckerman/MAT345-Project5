#include <iostream>     // cout, endl
#include <vector>       // vector
#include <numeric>      // inner_product
#include <ranges>       // views
#include <algorithm>    // transform
#include <cstdlib>      // atoi
#include <cmath>        // exp

#include "loader.cpp"
#include "prng.cpp"

using byte = std::byte;
template <typename T>
using vector = std::vector<T>;

struct WeightMatrix
{
    int numRows;
    int numCols;
    vector<vector<float>> weights;
    
    WeightMatrix(int r, int c) : numRows(r), numCols(c) 
    {
        weights = vector<vector<float>>(r, vector<float>(c));
        for (auto &i : weights)
        {
            for (auto &j : i) j = prng();
        }
    };
};

auto Sigmoid = [](float f) { return 1.0f / (1.0f + std::exp(-f)); };

int main(int argc, char** argv)
{
    int hiddenSize = std::atoi(argv[1]);
    int trainingSize = 60000;
    int imageSize = 784;
    
    const byte *const trainingLabels = LoadLabels("train-labels.idx1-ubyte");
    const byte *const trainingImages = LoadImages("train-images.idx3-ubyte");
    
    WeightMatrix W1(hiddenSize, imageSize + 1);
    WeightMatrix W2(10, hiddenSize + 1);
    
    // for (int i = 0; i < trainingSize; ++i)
    {
        vector<float> inputLayer(imageSize + 1, 1.0f);
        {
            for (int pixel = 1; pixel < imageSize + 1; ++pixel)
            {
                inputLayer[pixel] = (float)*trainingImages / 255.0f;
            }
        }
        
        vector<float> Y(10, 0.0f);
        {
            int label = (int)*trainingLabels;
            if (label == 0) label = 10;
            Y[label - 1] = 1.0f;
        }
        
        vector<float> hiddenLayer(hiddenSize + 1, 1.0f);
        {
            for (int i = 0; i < hiddenSize; ++i)
            {
                auto &row = W1.weights[i];
                hiddenLayer[i + 1] = std::inner_product(row.begin(), row.end(), inputLayer.begin(), 0.0f);
            }
            
            auto z = hiddenLayer | std::views::drop(1);
            std::ranges::transform(z, z.begin(), Sigmoid);
        }
        
        vector<float> outputLayer(10, 0.0f);
    }
    
    //byte *testLabels = LoadLabels("t10k-labels.idx1-ubyte");
    //byte *testImages = LoadImages("t10k-images.idx3-ubyte");
}