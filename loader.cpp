#include <fstream>      // ifstream    
#include <cstddef>      // byte

// @Source: https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
// @Note (jonathan): We don't need to worry about this implementation, I tested it and it works
auto ReverseInt(int i)
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

std::byte* LoadLabels(const char *filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw new std::exception();
    
    uint32_t endianCheck;
    file.read((char*)&endianCheck, sizeof(endianCheck));
    endianCheck = ReverseInt(endianCheck);
    if (endianCheck != 2049) throw new std::exception();
    
    uint32_t size;
    file.read((char*)&size, sizeof(size));
    size = ReverseInt(size);
    
    std::byte *data = new std::byte[size];
    file.read((char*)data, size);
    return data;
}

std::byte* LoadImages(const char *filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw new std::exception();
    
    uint32_t endianCheck;
    file.read((char*)&endianCheck, sizeof(endianCheck));
    endianCheck = ReverseInt(endianCheck);
    if (endianCheck != 2051) throw new std::exception();
    
    uint32_t size;
    file.read((char*)&size, sizeof(size));
    size = ReverseInt(size);
    
    uint32_t numRows;
    uint32_t numCols;
    file.read((char*)&numRows, sizeof(numRows));
    file.read((char*)&numCols, sizeof(numCols));
    numRows = ReverseInt(numRows);
    numCols = ReverseInt(numCols);
    if (numRows != 28 || numCols != 28) throw new std::exception();
    
    std::byte *data = new std::byte[size];
    file.read((char*)data, size);
    return data;
}