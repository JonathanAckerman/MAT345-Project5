#include <iostream>     // cout, endl
#include <vector>       // vector    

#include "loader.cpp"

int main()
{
    const std::byte *const trainingLabels = LoadLabels("train-labels.idx1-ubyte");
    const std::byte *const trainingImages = LoadImages("train-images.idx3-ubyte");
    
    int trainingSize = 60000;
    
    // @Note (jonathan): I'm getting decent performance here which is a bit surprising.
    //  My expectation is sizeof(float) * 10 * 60000 = 2,400,000 bytes
    std::vector<std::vector<float>> Y(trainingSize, std::vector<float>(10, 0.0f));
    auto labelIter = trainingLabels;
    for (int i = 0; i < trainingSize; ++i)
    {
        int label = (int)*labelIter;
        if (label == 0) label = 10;
        Y[i][label - 1] = 1.0f;
        ++labelIter;
    }
    
    //std::byte *testLabels = LoadLabels("t10k-labels.idx1-ubyte");
    //std::byte *testImages = LoadImages("t10k-images.idx3-ubyte");
    
}