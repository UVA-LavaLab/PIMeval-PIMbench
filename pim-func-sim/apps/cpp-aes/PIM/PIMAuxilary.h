#ifndef PIMAUXILARY_H
#define PIMAUXILARY_H

#include <vector>
#include <algorithm>
#include "libpimsim.h"

class PIMAuxilary {
public:
    std::vector<int> array;
    PimObjId pimObjId;
    PimAllocEnum allocType;
    unsigned numElements;
    unsigned bitsPerElements;
    PimDataType dataType;


    PIMAuxilary();
    PIMAuxilary(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElements, PimDataType dataType);
    PIMAuxilary(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElement, PimObjId ref, PimDataType dataType);
    PIMAuxilary(const PIMAuxilary* src); // Copy constructor
    ~PIMAuxilary();

    std::vector<int>* getArray();
    void setArray(const std::vector<int>& newArray);
    PimObjId getPimObjId() const;
    void setPimObjId(PimObjId id);
    int verifyArrayEquality(const std::vector<int>& otherArray) const;
};

// PIM functions to be added 
void pimShiftLeft(PIMAuxilary* x, int shiftAmount);
void pimShiftRight(PIMAuxilary* x, int shiftAmount);
void pimMul_(PIMAuxilary* src1, PIMAuxilary* src2, PIMAuxilary* dst);
void pimXor_(PIMAuxilary* src1, PIMAuxilary* src2, PIMAuxilary* dst);
void pimCopyDeviceToDevice(PIMAuxilary* src, PIMAuxilary* dst);




#endif // PIMAUXILARY_H
