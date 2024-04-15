#include "PIMAuxilary.h"
#include <cstdlib>
#include <iostream>


PIMAuxilary::PIMAuxilary() : pimObjId(0), allocType(PIM_ALLOC_V1), numElements(0), bitsPerElements(0), dataType(PIM_INT32) {
    // Constructor initialization
}

PIMAuxilary::PIMAuxilary(const PIMAuxilary* src) {
    this->pimObjId = pimAllocAssociated(src->allocType, src->numElements, src->bitsPerElements, src->pimObjId, src->dataType);
    if (this->pimObjId == -1) {
        std::cout << "Abort" << std::endl;
        abort();
    }
    this->array = *(new std::vector<int>(src->numElements));
    this->array.assign(src->array.begin(), src->array.end());  // Copy all elements from src to dest
    this->allocType = src->allocType;
    this->numElements = src->numElements;
    this->bitsPerElements = src->bitsPerElements;
    this->dataType = src->dataType;
}

PIMAuxilary::PIMAuxilary(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElements, PimDataType dataType) {
    this->pimObjId = pimAlloc(allocType, numElements, bitsPerElements, dataType);
    if (this->pimObjId == -1) {
        std::cout << "Abort" << std::endl;
        abort();
    }
    this->array = *(new std::vector<int>(numElements));
    this->allocType = allocType;
    this->numElements = numElements;
    this->bitsPerElements = bitsPerElements;
    this->dataType = dataType;
}

PIMAuxilary::PIMAuxilary(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElements, PimObjId ref, PimDataType dataType) {
    this->pimObjId = pimAllocAssociated(allocType, numElements, bitsPerElements, ref, dataType);
    if (this->pimObjId == -1) {
        std::cout << "Abort" << std::endl;
        abort();
    }
    this->array = std::vector<int>(numElements);
    this->allocType = allocType;
    this->numElements = numElements;
    this->bitsPerElements = bitsPerElements;
    this->dataType = dataType;
}


PIMAuxilary::~PIMAuxilary() {
    // Destructor - for cleanup, if needed
}

std::vector<int>* PIMAuxilary::getArray() {
    return &array;
}

void PIMAuxilary::setArray(const std::vector<int>& newArray) {
    array = newArray;
}

PimObjId PIMAuxilary::getPimObjId() const {
    return pimObjId;
}

void PIMAuxilary::setPimObjId(PimObjId id) {
    pimObjId = id;
}

int PIMAuxilary::verifyArrayEquality(const std::vector<int>& otherArray) const {
    return array == otherArray ? 1 : 0;
}

void pimShiftLeft(PIMAuxilary* x, int shiftAmount) // TODO: Add this function to the PIM API
{
    for (size_t i = 0; i < x->array.size(); ++i)
    {
        x->array[i] = x->array[i] << shiftAmount;
    }
}

void pimShiftRight(PIMAuxilary* x, int shiftAmount) // TODO: Add this function to the PIM API 
{
    for (size_t i = 0; i < x->array.size(); ++i)
    {
        x->array[i] = x->array[i] >> shiftAmount;
    }
}