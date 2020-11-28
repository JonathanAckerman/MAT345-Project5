#include <iostream>     // cout, endl

#include "loader.cpp"

int main()
{
    std::byte *trainingLabels = LoadLabels("train-labels.idx1-ubyte");
    std::byte *trainingImages = LoadImages("train-images.idx3-ubyte");
    std::byte *testLabels = LoadLabels("t10k-labels.idx1-ubyte");
    std::byte *testImages = LoadImages("t10k-images.idx3-ubyte");
    
}