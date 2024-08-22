// File: pimUtils.h
// PIMeval Simulator - Utilities
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef LAVA_PIM_UTILS_H
#define LAVA_PIM_UTILS_H

#include "libpimeval.h"
#include <string>
#include <queue>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>


namespace pimUtils
{
  std::string pimStatusEnumToStr(PimStatus status);
  std::string pimDeviceEnumToStr(PimDeviceEnum deviceType);
  std::string pimAllocEnumToStr(PimAllocEnum allocType);
  std::string pimCopyEnumToStr(PimCopyEnum copyType);
  std::string pimDataTypeEnumToStr(PimDataType dataType);
  unsigned getNumBitsOfDataType(PimDataType dataType);

  std::vector<bool> readBitsFromHost(void* src, uint64_t numElements, unsigned bitsPerElement);
  bool writeBitsToHost(void* dest, const std::vector<bool>& bits);

  //! @class  threadWorker
  //! @brief  Thread worker base class
  class threadWorker {
  public:
    threadWorker() {}
    virtual ~threadWorker() {}
    virtual void execute() = 0;
  };

  //! @class  threadPool
  //! @brief  Thread pool that runs multiple workers in threads
  class threadPool {
  public:
    threadPool(size_t numThreads);
    ~threadPool();
    void doWork(const std::vector<pimUtils::threadWorker*>& workers);
  private:
    void workerThread();

    std::vector<std::thread> m_threads;
    std::queue<threadWorker*> m_workers;
    std::mutex m_mutex;
    std::condition_variable m_cond;
    bool m_terminate;
    std::atomic<size_t> m_workersRemaining;
  };

}

#endif

