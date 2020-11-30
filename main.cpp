#include <iostream>     // cout, endl
#include <vector>       // vector
#include <numeric>      // inner_product
#include <ranges>       // views
#include <algorithm>    // transform
#include <functional>   // minus
#include <cstdlib>      // atoi
#include <cmath>        // exp

#include "loader.cpp"
#include "prng.cpp"

using byte = std::byte;
template <typename T>
using vector = std::vector<T>;

vector<float> operator-(vector<float> lhs, vector<float> rhs)
{
    vector<float> result(lhs.size());
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), std::minus<float>());
    return result;
}

struct WeightMatrix
{
    int numRows;
    int numCols;
    vector<vector<float>> weights;
    
    WeightMatrix(int r, int c) : numRows(r), numCols(c) 
    {
        weights = vector<vector<float>>(r, vector<float>(c));
    };
    
    void Initialize()
    {
        for (auto &i : weights)
        {
            for (auto &j : i) j = prng();
        }
    }
    
    WeightMatrix Transpose()
    {
        WeightMatrix result(numCols, numRows);
        for (int row = 0; row < numRows; ++row)
        {
            for (int col = 0; col < numCols; ++col)
            {
                result.weights[col][row] = weights[row][col];
            }
        }
        return result;
    }
    
    WeightMatrix NoBias()
    {
        WeightMatrix result(numRows, numCols - 1);
        for (int row = 0; row < numRows; ++row)
        {
            for (int col = 0; col < numCols - 1; ++col)
            {
                result.weights[row][col] = weights[row][col + 1];
            }
        }
        return result; 
    }
};

auto Sigmoid = [](float f) { return 1.0f / (1.0f + std::exp(-f)); };
auto derivativeSigmoid = [](float f) { return Sigmoid(f) * (1.0f - Sigmoid(f)); };

void FeedForward(const vector<float> curLayer, vector<float>& nextLayer, const WeightMatrix Wk, const bool hasBias)
{
    int size = hasBias ? nextLayer.size() - 1 : nextLayer.size();
    
    for (int i = 0; i < size; ++i)
    {
        auto &row = Wk.weights[i];
        int index = hasBias ? i + 1 : i;
        nextLayer[index] = std::inner_product(row.begin(), row.end(), curLayer.begin(), 0.0f);
    }

    auto z = hasBias ?
        nextLayer | std::views::drop(1):
        nextLayer | std::views::drop(0);
    std::ranges::transform(z, z.begin(), Sigmoid);
}

int main(int argc, char** argv)
{
    int hiddenSize = std::atoi(argv[1]);
    int trainingSize = 60000;
    int imageSize = 784;
    
    const byte *const trainingLabels = LoadLabels("train-labels.idx1-ubyte");
    const byte *const trainingImages = LoadImages("train-images.idx3-ubyte");
    
    WeightMatrix W1(hiddenSize, imageSize + 1);
    WeightMatrix W2(10, hiddenSize + 1);
    W1.Initialize();
    W2.Initialize();
    
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
        FeedForward(inputLayer, hiddenLayer, W1, true);
        
        vector<float> outputLayer(10, 0.0f);
        FeedForward(hiddenLayer, outputLayer, W2, false);
        
        vector<float> error3 = outputLayer - Y;
        WeightMatrix W2_Prime = W2.NoBias().Transpose();
    }
    
    //byte *testLabels = LoadLabels("t10k-labels.idx1-ubyte");
    //byte *testImages = LoadImages("t10k-images.idx3-ubyte");
}