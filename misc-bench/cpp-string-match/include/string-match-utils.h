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

#endif