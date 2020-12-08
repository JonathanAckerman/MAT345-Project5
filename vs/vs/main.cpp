#include <iostream>     // cout, endl
#include <vector>       // vector
#include <numeric>      // inner_product
#include <algorithm>    // transform
#include <functional>   // minus, multiplies
#include <cstdlib>      // atoi
#include <cmath>        // exp, log
#include <utility>      // tuple

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

    WeightMatrix& operator*=(const float f)
    {
        for (auto &i : weights)
        {
            for (auto &j : i) j *= f;
        }
        return *this;
    }

    WeightMatrix& operator-=(WeightMatrix& rhs)
    {
        if (numRows != rhs.numRows || numCols != rhs.numCols)
            throw new std::exception();

        for (int i = 0; i < numRows; ++i)
        {
            for (int j = 0; j < numCols; ++j)
            {
                weights[i][j] -= rhs.weights[i][j];
            }
        }

        return *this;
    }
};


auto Sigmoid = [](float f) { return 1.0f / (1.0f + std::exp(-f)); };
auto derivativeSigmoid = [](float f) { return Sigmoid(f) * (1.0f - Sigmoid(f)); };

vector<float> FeedForward(const WeightMatrix Wk, const vector<float> curLayer)
{
    vector<float> z;
    for (int i = 0; i < Wk.numRows; ++i)
    {
        auto& row = Wk.weights[i];
        z.push_back(std::inner_product(row.begin(), row.end(), curLayer.begin(), 0.0f));
    }
    return z;
}

float Cost(const vector<float> predicted, const vector<float> expected)
{
    vector<float> logPred = predicted;
    std::transform(predicted.begin(), predicted.end(), logPred.begin(), [](float f) { return std::log(f); });
    float lhs = std::inner_product(expected.begin(), expected.end(), logPred.begin(), 0.0f);

    vector<float> oneVector = vector<float>(10, 1.0f);
    vector<float> rhs1 = oneVector - expected;
    vector<float> rhs2 = oneVector - predicted;
    std::transform(rhs2.begin(), rhs2.end(), rhs2.begin(), [](float f) { return std::log(f); });
    float rhs = std::inner_product(rhs1.begin(), rhs1.end(), rhs2.begin(), 0.0f);

    return lhs + rhs;
}

int main(int argc, char** argv)
{
    int hiddenSize = std::atoi(argv[1]);
    int trainingSize = 60000;
    int testingSize = 10000;
    int imageSize = 784;
    int numItr = 1;
    float n = 0.001f;

    //auto startTraining = std::chrono::high_resolution_clock::now();
    
    vector<float> trainingImages;
    vector<vector<float>> trainingLabels;
    {
        const byte* const trainingLabelBytes = LoadLabels("train-labels.idx1-ubyte");
        const byte* const trainingImageBytes = LoadImages("train-images.idx3-ubyte");

        for (int i = 0; i < trainingSize * imageSize; ++i)
        {
            float f = (float)*(trainingImageBytes + i) / 255.f;
            trainingImages.push_back(f);
        }
        for (int i = 0; i < trainingSize; ++i)
        {
            int label = (int)*(trainingLabelBytes + i);
            vector<float> v(10, 0.0f);
            v[label] = 1.0f;
            trainingLabels.push_back(v);
        }

        delete trainingImageBytes;
        delete trainingLabelBytes;
    }
    
    WeightMatrix W1(hiddenSize, imageSize + 1);
    WeightMatrix W2(10, hiddenSize + 1);
    W1.Initialize();
    W2.Initialize();

    auto pixelIter = trainingImages.begin();
    for (unsigned t = 0; t < numItr; ++t)
    {
        WeightMatrix gW1(hiddenSize, imageSize + 1);
        WeightMatrix gW2(10, hiddenSize + 1);
        gW1.Initialize(0.0f);
        gW2.Initialize(0.0f);

        for (int image = 0; image < trainingSize; ++image)
        {
            vector<float> inputLayer(pixelIter, pixelIter + imageSize);
            inputLayer.insert(inputLayer.begin(), 1.0f);

            vector<float> z2 = FeedForward(W1, inputLayer);
            vector<float> a2 = z2;
            std::transform(a2.begin(), a2.end(), a2.begin(), Sigmoid);
            a2.insert(a2.begin(), 1.0f);

            vector<float> z3 = FeedForward(W2, a2);
            vector<float> a3 = z3;
            std::transform(a3.begin(), a3.end(), a3.begin(), Sigmoid);

            //float c = Cost(a3, trainingLabels[image]);
            //std::cout << "t: " << t << " image: "<< image << ", Cost(): " << c << " n: " << n << std::endl;
            
            // Back Propagation
            {
                vector<float> d3 = a3 - trainingLabels[image];
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

        gW1 *= n;
        gW2 *= n;
        W1 -= gW1;
        W2 -= gW2;

        std::cout << ".";
    }

    //auto endTraining = std::chrono::high_resolution_clock::now();

    const byte* const testingLabels = LoadLabels("t10k-labels.idx1-ubyte");
    const byte* const testingImages = LoadImages("t10k-images.idx3-ubyte");

    //auto startTesting = std::chrono::high_resolution_clock::now();

    vector<std::tuple<int, int, float>> testPredictions;
    auto dataIter = testingImages;
    for (int image = 0; image < testingSize; ++image)
    {
        vector<float> inputLayer(imageSize);
        for (int pixel = 0; pixel < imageSize + 0; ++pixel)
        {
            inputLayer[pixel] = (float)*dataIter / 255.0f;
            ++dataIter;
        }
        inputLayer.insert(inputLayer.begin(), 1.0f);

        vector<float> Y(10, 0.0f);
        int label = (int)*(testingLabels + image * sizeof(byte));
        Y[label] = 1.0f;

        vector<float> z2 = FeedForward(W1, inputLayer);
        vector<float> a2 = z2;
        std::transform(a2.begin(), a2.end(), a2.begin(), Sigmoid);
        a2.insert(a2.begin(), 1.0f);

        vector<float> z3 = FeedForward(W2, a2);
        vector<float> a3 = z3;
        std::transform(a3.begin(), a3.end(), a3.begin(), Sigmoid);

        auto maxElem = std::max_element(a3.begin(), a3.end());
        int predLabel = std::distance(a3.begin(), maxElem);
        testPredictions.push_back(std::tuple(predLabel, label, *maxElem));
    }

    //auto endTesting = std::chrono::high_resolution_clock::now();

    //auto trainDur = std::chrono::duration_cast<std::chrono::seconds>(endTraining - startTraining);
    //auto testDur = std::chrono::duration_cast<std::chrono::seconds>(endTesting - startTesting);

    //std::cout << "trainDur: " << trainDur.count() << std::endl;
    //std::cout << "testDur: " << testDur.count() << std::endl;
    
    auto activated = [](std::tuple<int, int, float> p) { return std::get<2>(p) > 0.5f; };
    std::cout << "Count >0.5: " << std::count_if(testPredictions.begin(), testPredictions.end(), activated);
    
    auto matched = [](std::tuple<int, int, float> p) { return std::get<0>(p) == std::get<1>(p); };
    int accurate = 0;
    for (int i = 0; i < testingSize; ++i)
    {
        if (activated(testPredictions[i]) && matched(testPredictions[i]))
        {
            ++accurate;
        }
    }
    std::cout << " Accuracy: " << accurate << std::endl;
    std::cout << std::endl;

    std::ofstream w1File;
    w1File.open("W1.csv");
    for (auto i : W1.weights)
    {
        for (auto j : i)
        {
            w1File << j << ", ";
        }
        w1File << std::endl;
    }
    w1File.close();

    std::ofstream w2File;
    w2File.open("W2.csv");
    for (auto i : W2.weights)
    {
        for (auto j : i)
        {
            w2File << j << ", ";
        }
        w2File << std::endl;
    }
    w2File.close();
}