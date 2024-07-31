/**
 * @file transitive-closure.hpp
 * @brief Helper file that provides a CSV reading function for the PIM, CPU, and GPU benchmarks
 *
 */

#include <fstream>
#include <sstream>
#include <iostream>

// max edge value acting as INF, can be set to a much greater value
#define MAX_EDGE_VALUE 10000 


bool readCSV(const std::string &filename, std::vector<int> &inputList, int &numVertices) 
{
  std::ifstream file(filename);
  if (!file.is_open()) 
  {
    std::cerr << "Error opening file: " << filename << std::endl;
    return false;
  }

  std::string line;
  if (std::getline(file, line)) {
    std::istringstream iss(line);
    iss >> numVertices;
  } 
  else 
  {
    std::cerr << "Error reading number of vertices from file: " << filename << std::endl;
    return false;
  }

  while (std::getline(file, line)) 
  {
    std::istringstream iss(line);
    std::string value;

    while (std::getline(iss, value, ',')) {
      if (value.find("inf") != std::string::npos) {
        inputList.push_back(MAX_EDGE_VALUE);
      } 
      else 
      {
        try 
        {
          inputList.push_back(std::stoi(value));
        } catch (const std::invalid_argument& e) {
          std::cerr << "Invalid value in file: " << value << std::endl;
          return false;
        } catch (const std::out_of_range& e) {
          std::cerr << "Value out of range in file: " << value << std::endl;
          return false;
        }
      }
    }
  }  

  file.close();
  return true;
}
