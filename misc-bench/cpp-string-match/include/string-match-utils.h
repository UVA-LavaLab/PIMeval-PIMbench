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

void stringMatchCpu(std::vector<std::string>& needles, std::string& haystack, std::vector<int>& matches) {
  for(uint64_t needleIdx = 0; needleIdx < needles.size(); ++needleIdx) {
    size_t pos = haystack.find(needles[needleIdx], 0);

    while (pos != std::string::npos) {
        matches[pos] = std::max((unsigned long) matches[pos], needleIdx + 1);
        pos = haystack.find(needles[needleIdx], pos + 1);
    }
  }
}

#endif