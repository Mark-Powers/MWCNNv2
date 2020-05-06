#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>

#include "image_io.hh"

namespace fs = std::filesystem;

int main(int argc, char **argv) {
  const fs::path input_directory{argc > 1 ? argv[1] : fs::current_path()};

  int total_dng = 0;

  std::vector<std::string> list = {
//      "62CL_20150215_180549_095",
//      "6G7M_20150403_165723_505",
//      "5a9e_20150404_122216_530",
//      "c1b1_20150311_204258_018",
//      "0830_20151101_123037_908",
//      "0155_20160903_150944_775",
//      "0030_20151006_135654_478",
//      "47L8_20150517_182450_843",
//      "4WBR_20150516_112748_501",
//      "0006_20160726_142655_865",
//      "0132_20160917_184610_200",
//      "0043_20161005_164046_315",
//      "c1b1_20150408_154240_395",
//      "0043_20160922_133339_966",
//      "0032_20160921_125350_046",
//      "33TJ_20150705_191438_366",
//      "JN34_20150319_183334_317",
//      "6G7M_20150421_121002_835",
//      "0919_20150905_182209_361",
//      "0830_20151108_120206_194"
  };

  auto t1 = std::chrono::high_resolution_clock::now();
  for (auto entry = fs::recursive_directory_iterator(input_directory);
       entry != fs::recursive_directory_iterator();
       ++entry) {
    if (fs::is_regular_file(entry->path()) && entry->path().extension() == ".dng") {
      auto f = entry->path();
      auto p = entry->path().parent_path();
      std::string filepath = f.string();
      std::string parent = p.string();
      std::string dng = p.stem().string();

      filepath.erase(remove(filepath.begin(), filepath.end(), '\"'), filepath.end());
      parent.erase(remove(parent.begin(), parent.end(), '\"'), parent.end());
      dng.erase(remove(dng.begin(), dng.end(), '\"'), dng.end());

      std::cout << dng << std::endl;
      if (list.empty() || std::find(list.begin(), list.end(), dng) != list.end()) {
        auto time = process(filepath, parent);
        ++total_dng;
      }
    }
  }

  auto t2 = std::chrono::high_resolution_clock::now();

  auto total_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

  std::cout << "Total DNGs: " << total_dng << std::endl;
  std::cout << "Total FPS: " << total_dng / total_time.count() << std::endl;
  std::cout << "Total Time: " << total_time.count() / 3600.f << std::endl;
}