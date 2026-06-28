#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "schema/model_generated.h"

namespace {

std::vector<uint8_t> ReadFile(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.good()) return {};
  const auto size = file.tellg();
  std::vector<uint8_t> data(static_cast<size_t>(size));
  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char *>(data.data()), size);
  return data;
}

int DumpOps(const std::string &path) {
  const auto data = ReadFile(path);
  if (data.empty()) {
    std::cerr << "model not found or empty: " << path << std::endl;
    return 2;
  }
  flatbuffers::Verifier verifier(data.data(), data.size());
  if (!mindspore::schema::VerifyMetaGraphBuffer(verifier)) {
    std::cerr << "not a valid MetaGraph flatbuffer: " << path << std::endl;
    return 3;
  }
  const auto *graph = mindspore::schema::GetMetaGraph(data.data());
  const auto *nodes = graph == nullptr ? nullptr : graph->nodes();
  if (nodes == nullptr) {
    std::cerr << "nodes=null: " << path << std::endl;
    return 4;
  }
  std::map<std::string, int> counts;
  for (uint32_t i = 0; i < nodes->size(); ++i) {
    const auto *node = nodes->Get(i);
    const auto *prim = node == nullptr ? nullptr : node->primitive();
    const auto type = prim == nullptr ? mindspore::schema::PrimitiveType_NONE : prim->value_type();
    const char *name = mindspore::schema::EnumNamePrimitiveType(type);
    counts[name == nullptr || name[0] == '\0' ? "UNKNOWN" : name] += 1;
  }

  std::vector<std::pair<std::string, int>> sorted(counts.begin(), counts.end());
  std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) {
    if (a.second != b.second) return a.second > b.second;
    return a.first < b.first;
  });

  std::cout << "model=" << path << std::endl;
  std::cout << "model_size=" << data.size() << std::endl;
  std::cout << "node_count=" << nodes->size() << std::endl;
  std::cout << "op_type_count=" << sorted.size() << std::endl;
  for (const auto &item : sorted) {
    std::cout << item.first << "," << item.second << std::endl;
  }
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: mslite_op_dump <model.ms>" << std::endl;
    return 1;
  }
  return DumpOps(argv[1]);
}
