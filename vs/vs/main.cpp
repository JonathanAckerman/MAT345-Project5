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
    
    void Initialize(float zero)
    {
        for (auto &i : weights)
        {
            for (auto &j : i) j = 0.0f;
        }
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

int main(int argc, char** argv)
{
    int hiddenSize = std::atoi(argv[1]);
    int trainingSize = 60000;
    int imageSize = 784;
    int numItr = 1;
    float n = 0.1f;
    
    const byte *const trainingLabels = LoadLabels("train-labels.idx1-ubyte");
    const byte *const trainingImages = LoadImages("train-images.idx3-ubyte");
    vector<vector<float>> tI = vector<vector<float>>(trainingSize, vector<float>(imageSize, 0.0f));
    for (int i = 0; i < trainingSize; ++i)
    {
        for (int j = 0; j < imageSize; ++j)
        {
            tI[i][j] = (float)*(trainingImages + i * imageSize + j);
        }
    }
    delete trainingImages;
    
    WeightMatrix W1(hiddenSize, imageSize + 1);
    WeightMatrix W2(10, hiddenSize + 1);
    W1.Initialize();
    W2.Initialize();
    
    for (unsigned t = 0; t < numItr; ++t)
    {
        WeightMatrix gW1(hiddenSize, imageSize + 1);
        WeightMatrix gW2(10, hiddenSize + 1);
        gW1.Initialize(0.0f);
        gW2.Initialize(0.0f);
        
        for (int image = 0; image < trainingSize; ++image)
        {
            vector<float> inputLayer(imageSize);
            for (int pixel = 0; pixel < imageSize + 0; ++pixel)
            {
                inputLayer[pixel] = tI[image][pixel] / 255.0f;
            }
            inputLayer.insert(inputLayer.begin(), 1.0f);
            
            vector<float> Y(10, 0.0f);
            {
                int label = (int)*(trainingLabels + image * sizeof(byte));
                Y[label] = 1.0f;
            }
            
            vector<float> z2 = FeedForward(W1, inputLayer);
            vector<float> a2 = z2;
            std::transform(a2.begin(), a2.end(), a2.begin(), Sigmoid);
            a2.insert(a2.begin(), 1.0f);
            
            vector<float> z3 = FeedForward(W2, a2);
            vector<float> a3 = z3;
            std::transform(a3.begin(), a3.end(), a3.begin(), Sigmoid);

            // Back Propagation
            {
                vector<float> d3 = a3 - Y;
                WeightMatrix W2_Prime = W2.NoBias().Transpose();
                vector<float> d2;
                for (auto row : W2_Prime.weights)
                {
                    d2.push_back(std::inner_product(row.begin(), row.end(), d3.begin(), 0.0f));
                }
                vector<float> z2_Prime = z2;
                {
                    std::transform(z2_Prime.begin(), z2_Prime.end(), z2_Prime.begin(), derivativeSigmoid);
                }
                std::transform(d2.begin(), d2.end(), z2_Prime.begin(), d2.begin(), std::multiplies<float>());
                
                for (int i = 0; i < hiddenSize; ++i)
                {
                    for (int j = 0; j < imageSize + 1; ++j)
                    {
                        gW1.weights[i][j] += d2[i] * inputLayer[j];
                    }
                }
                
                for (int i = 0; i < 10; ++i)
                {
                    for (int j = 0; j < hiddenSize + 1; ++j)
                    {
                        gW2.weights[i][j] += d3[i] * a2[j];
                    }
                }
            }
        }

        std::cout << "Done with the loop" << std::endl;
        
        for (int i = 0; i < gW1.numRows; ++i)
        {
            for (int j = 0; j < gW1.numCols; ++j)
            {
                gW1.weights[i][j] /= (float)trainingSize;
            }
        }
        
        for (int i = 0; i < gW2.numRows; ++i)
        {
            for (int j = 0; j < gW2.numCols; ++j)
            {
                gW2.weights[i][j] /= (float)trainingSize;
            }
        }
        
        WeightMatrix *temp = n * gW1;
        WeightMatrix *newW1 = W1 - *temp;
        W1 = *newW1;
        delete temp;
        delete newW1;
        
        WeightMatrix *temp2 = n * gW2;
        WeightMatrix *newW2 = W2 - *temp2;
        W2 = *newW2;
        delete temp2;
        delete newW2;
    }

    std::cout << "DONE! w1[0][0]: " <<  W1.weights[0][0];
    
    
    //byte *testLabels = LoadLabels("t10k-labels.idx1-ubyte");
    //byte *testImages = LoadImages("t10k-images.idx3-ubyte");
}