#ifndef STRING_MATCH_UTILS_H
#define STRING_MATCH_UTILS_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>

std::vector<std::string> get_needles_from_file(const std::string& dataset_prefix, const std::string& keys_filename) {
  std::string keys_input_file = dataset_prefix + keys_filename;
  std::ifstream keys_file(keys_input_file);
  if (!keys_file) {
    return {};
  }
  
  std::vector<std::string> needles;
  std::string line;
  while (std::getline(keys_file, line)) {
      needles.push_back(line);
  }
  
  return needles;
}

std::string get_text_from_file(const std::string& dataset_prefix, const std::string& text_filename) {
  std::string text_input_file = dataset_prefix + text_filename;
  
  std::ifstream text_file(text_input_file);
  if (!text_file) {
    return "";
  }

  std::ostringstream text_file_oss;
  text_file_oss << text_file.rdbuf();
  return text_file_oss.str();
}

void string_match_cpu(std::vector<std::string>& needles, std::string& haystack, std::vector<int>& matches) {
  for(uint64_t needle_idx = 0; needle_idx < needles.size(); ++needle_idx) {
    size_t pos = haystack.find(needles[needle_idx], 0);

    while (pos != std::string::npos) {
        matches[pos] = std::max((unsigned long) matches[pos], needle_idx + 1);
        pos = haystack.find(needles[needle_idx], pos + 1);
    }
  }
}

#endif