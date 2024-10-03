#ifndef PIMAUXILARY_H
#define PIMAUXILARY_H

#include <vector>
#include <algorithm>
#include <cstdint>
#include "libpimeval.h"

class PIMAuxilary {
public:
    std::vector<uint8_t> array;
    PimObjId pimObjId;
    PimAllocEnum allocType;
    unsigned numElements;
    PimDataType dataType;


    PIMAuxilary();
    PIMAuxilary(PimAllocEnum allocType, unsigned numElements, PimDataType dataType);
    PIMAuxilary(PimAllocEnum allocType, unsigned numElements, PimObjId ref, PimDataType dataType);
    PIMAuxilary(const PIMAuxilary* src); // Copy constructor
    ~PIMAuxilary();

    std::vector<uint8_t>* getArray();
    void setArray(const std::vector<uint8_t>& newArray);
    PimObjId getPimObjId() const;
    void setPimObjId(PimObjId id);
    int verifyArrayEquality(const std::vector<uint8_t>& otherArray) const;
};

#endif // PIMAUXILARY_H
