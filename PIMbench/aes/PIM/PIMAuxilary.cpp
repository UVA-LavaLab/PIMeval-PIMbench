#include "PIMAuxilary.h"
#include <cstdlib>
#include <iostream>
#include <cassert>

PIMAuxilary::PIMAuxilary() : pimObjId(0), allocType(PIM_ALLOC_AUTO), numElements(0), dataType(PIM_UINT8) {
    // Constructor initialization
}

PIMAuxilary::PIMAuxilary(const PIMAuxilary* src) {
    this->pimObjId = pimAllocAssociated(src->pimObjId, src->dataType);
    if (this->pimObjId == -1) {
        std::cout << "Abort" << std::endl;
        abort();
    }
    this->array = *(new std::vector<uint8_t>(src->numElements));
    this->array.assign(src->array.begin(), src->array.end());  // Copy all elements from src to dest
    this->allocType = src->allocType;
    this->numElements = src->numElements;
    this->dataType = src->dataType;
}

PIMAuxilary::PIMAuxilary(PimAllocEnum allocType, unsigned numElements, PimDataType dataType) {
    this->pimObjId = pimAlloc(allocType, numElements, dataType);
    if (this->pimObjId == -1) {
        std::cout << "Abort" << std::endl;
        abort();
    }
    this->array = *(new std::vector<uint8_t>(numElements));
    this->allocType = allocType;
    this->numElements = numElements;
    this->dataType = dataType;
}

PIMAuxilary::PIMAuxilary(PimAllocEnum allocType, unsigned numElements, PimObjId ref, PimDataType dataType) {
    this->pimObjId = pimAllocAssociated(ref, dataType);
    if (this->pimObjId == -1) {
        std::cout << "Abort" << std::endl;
        abort();
    }
    this->array = std::vector<uint8_t>(numElements);
    this->allocType = allocType;
    this->numElements = numElements;
    this->dataType = dataType;
}

PIMAuxilary::~PIMAuxilary() {
    // Destructor - for cleanup, if needed
}

std::vector<uint8_t>* PIMAuxilary::getArray() {
    return &array;
}

void PIMAuxilary::setArray(const std::vector<uint8_t>& newArray) {
    array = newArray;
}

PimObjId PIMAuxilary::getPimObjId() const {
    return pimObjId;
}

void PIMAuxilary::setPimObjId(PimObjId id) {
    pimObjId = id;
}

int PIMAuxilary::verifyArrayEquality(const std::vector<uint8_t>& otherArray) const {
    return array == otherArray ? 1 : 0;
}

void pimShiftLeft(PIMAuxilary* x, int shiftAmount) { // TODO: Add this function to the PIM API
    for (size_t i = 0; i < x->array.size(); ++i)
    {
        x->array[i] = x->array[i] << shiftAmount;
    }
}

void pimShiftRight(PIMAuxilary* x, int shiftAmount) { // TODO: Add this function to the PIM API 
    for (size_t i = 0; i < x->array.size(); ++i)
    {
        x->array[i] = x->array[i] >> shiftAmount;
    }
}

void pimMul_(PIMAuxilary* src1, PIMAuxilary* src2, PIMAuxilary* dst) {
    PimStatus status; 
    status = pimCopyDeviceToHost(src1->pimObjId, (void*)src1->array.data()); 
    status = pimCopyDeviceToHost(src2->pimObjId, (void*)src2->array.data()); 
    for (size_t i = 0; i < dst->array.size(); ++i)
    {
        dst->array[i] = (src1->array[i] * src2->array[i]) % 256;
    }
    status = pimCopyHostToDevice((void*)dst->array.data(), dst->pimObjId); 
    assert(status == PIM_OK);
    int PimObjId = -1;
    status = pimMul(PimObjId, PimObjId, PimObjId); // TODO: Debug Xor
}

void pimXor_(PIMAuxilary* src1, PIMAuxilary* src2, PIMAuxilary* dst) {
    PimStatus status; 
    status = pimCopyDeviceToHost(src1->pimObjId, (void*)src1->array.data()); 
    status = pimCopyDeviceToHost(src2->pimObjId, (void*)src2->array.data()); 
    for (size_t i = 0; i < dst->array.size(); i++)
    {
        dst->array[i] = (src1->array[i] ^ src2->array[i]) % 256;
    }
    status = pimCopyHostToDevice((void*)dst->array.data(), dst->pimObjId); 
    assert(status == PIM_OK);
    int PimObjId = -1;

    // status = pimXor(PimObjId, PimObjId, PimObjId); // TODO: Debug Xor
    assert(status == PIM_OK);
}

void pimCopyDeviceToDevice(PIMAuxilary* src, PIMAuxilary* dst) { 
    PimStatus status; 
    status = pimCopyDeviceToHost(src->pimObjId, src->array.data());
    for(int i = 0; i < dst->array.size(); i++) {
            dst->array[i] = src->array[i];
    }
    status = pimCopyHostToDevice(dst->array.data(), dst->pimObjId);
    assert(status == PIM_OK);

}
