#ifndef STRING_MATCH_UTILS_H
#define STRING_MATCH_UTILS_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>

std::vector<std::string> getNeedlesFromFile(const std::string& keysInputFile) {
  std::ifstream keysFile(keysInputFile);
  if (!keysFile) {
    return {};
  }
  
  std::vector<std::string> needles;
  std::string line;
  while (std::getline(keysFile, line)) {
    // As the output is formatted as the 32 bit index of the needle for matches, there cannot be more than INT_MAX needles
    if(needles.size() == std::numeric_limits<int>::max()) {
      std::cerr << "Error: Too many needles/keys, can only process up to " << std::numeric_limits<int>::max() << " needles/keys" << std::endl;
      exit(1);
    }

    needles.push_back(line);
  }
  
  return needles;
}

std::string getTextFromFile(const std::string& textInputFile) {
  std::ifstream textFile(textInputFile);
  if (!textFile) {
    return "";
  }

  std::ostringstream textFileOss;
  textFileOss << textFile.rdbuf();
  return textFileOss.str();
}

void hammingStringMatchCpu(std::vector<std::string>& needles, std::string& haystack, uint64_t maxHammingDistance, std::vector<int>& matches) {
  for(uint64_t needleIdx = 0; needleIdx < needles.size(); ++needleIdx) {
    const std::string_view needle = needles[needleIdx];
    if(needle.size() > haystack.size()) {
      continue;
    }
    for(uint64_t haystackIdx = 0; haystackIdx <= haystack.size()-needle.size(); ++haystackIdx) {
      uint64_t mismatches = 0;
      for(uint64_t charIdx = 0; charIdx < needle.size(); ++charIdx) {
        if(haystack[haystackIdx + charIdx] != needle[charIdx]) {
          ++mismatches;
        }
      }
      if(mismatches <= maxHammingDistance) {
        matches[haystackIdx] = std::max(matches[haystackIdx], static_cast<int>(1 + needleIdx));
      }
    }
  }
}

#endif