#include <iostream>     // cout, endl
#include <vector>       // vector
#include <numeric>      // inner_product
#include <ranges>       // views
#include <algorithm>    // transform
#include <functional>   // minus, multiplies
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
    std::transform(
        lhs.begin(), 
        lhs.end(), 
        rhs.begin(), 
        result.begin(), 
        std::minus<float>()
    );
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

    // copy c'tor
    WeightMatrix(WeightMatrix const& origW)
    {
      numRows = origW.numRows;
      numCols = origW.numCols;
      weights = origW.weights;
    }
    
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

vector<float> FeedForward(const WeightMatrix Wk, const vector<float> curLayer)
{
    vector<float> z(Wk.numRows);
    for (int i = 0; i < Wk.numRows; ++i)
    {
        auto &row = Wk.weights[i];
        z[i] = std::inner_product(row.begin(), row.end(), curLayer.begin(), 0.0f);
    }
    return z;
}

// scalar product of mat
WeightMatrix* operator*(float lhs, WeightMatrix& rhs)
{
  WeightMatrix* result = new WeightMatrix(rhs.numRows, rhs.numCols);

  for (auto& i : result->weights)
  {
    for (auto& j : i) j *= lhs;
  }

  return result;
}

// weight mat subtraction
WeightMatrix* operator-(WeightMatrix& lhs, WeightMatrix& rhs)
{
  WeightMatrix* result = new WeightMatrix(rhs.numRows, rhs.numCols);

  for (int i = 0; i < result->numRows; ++i)
    result->weights[i] = lhs.weights[i] - rhs.weights[i];

  return result;
}

// grad des
void GradDes(WeightMatrix& W, float n, unsigned numItr)
{
  WeightMatrix* oldW = new WeightMatrix(W);

  // W(t+1) = W(t) - n*gradW(t)
  for (unsigned i = 0; i < numItr; ++i)
  {
    // TODO: gradW();
    WeightMatrix* gradW = /* gradW(); */
      new WeightMatrix(W.numRows, W.numCols); // placeholder
    WeightMatrix* step = n * *gradW;
    WeightMatrix* newW = *oldW - *step;

    // clear
    delete gradW;
    delete step;
    delete oldW;

    oldW = newW;
  }
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

    // store ini W for grad des
    WeightMatrix iniW1(W1);
    WeightMatrix iniW2(W2);
    
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
        
        vector<float> z2 = FeedForward(W1, inputLayer);
        vector<float> a2(z2.size() + 1, 1.0f);
        {
            auto view = a2 | std::views::drop(1);
            std::ranges::transform(view, view.begin(), Sigmoid);
        }
        
        vector<float> z3 = FeedForward(W2, a2);
        vector<float> a3 = z3;
        std::transform(a3.begin(), a3.end(), a3.begin(), Sigmoid);

        // Back Propagation
        {
            vector<float> d3 = a3 - Y;
            WeightMatrix W2_Prime = W2.NoBias().Transpose();
            vector<float> d2(W2_Prime.numRows);
            for (auto row : W2_Prime.weights)
            {
                d2.push_back(std::inner_product(row.begin(), row.end(), d3.begin(), 0.0f));
            }
            vector<float> z2_Prime = z2;
            {
                std::transform(z2_Prime.begin(), z2_Prime.end(), z2_Prime.begin(), derivativeSigmoid);
            }
            std::transform(d2.begin(), d2.end(), z2_Prime.begin(), d2.begin(), std::multiplies<float>());
        }
    }
    
    //byte *testLabels = LoadLabels("t10k-labels.idx1-ubyte");
    //byte *testImages = LoadImages("t10k-images.idx3-ubyte");
}