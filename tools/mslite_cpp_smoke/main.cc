#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "include/api/context.h"
#include "include/api/data_type.h"
#include "include/api/model.h"
#include "include/api/status.h"
#include "include/api/types.h"

namespace {

std::string ShapeText(const std::vector<int64_t> &shape) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) oss << ",";
    oss << shape[i];
  }
  oss << "]";
  return oss.str();
}

std::string DataTypeText(mindspore::DataType type) {
  switch (type) {
    case mindspore::DataType::kNumberTypeInt32:
      return "int32";
    case mindspore::DataType::kNumberTypeInt64:
      return "int64";
    case mindspore::DataType::kNumberTypeFloat32:
      return "float32";
    case mindspore::DataType::kNumberTypeFloat16:
      return "float16";
    case mindspore::DataType::kNumberTypeUInt8:
      return "uint8";
    default:
      return "type_" + std::to_string(static_cast<int>(type));
  }
}

int64_t ElementCount(const std::vector<int64_t> &shape) {
  if (shape.empty()) return 0;
  int64_t count = 1;
  for (auto dim : shape) {
    if (dim <= 0) return 0;
    count *= dim;
  }
  return count;
}

bool HasDynamicDim(const std::vector<int64_t> &shape) {
  for (auto dim : shape) {
    if (dim <= 0) return true;
  }
  return false;
}

std::vector<int64_t> TargetShape(const mindspore::MSTensor &tensor, size_t index, size_t input_count) {
  const auto shape = tensor.Shape();
  if (!HasDynamicDim(shape) && !shape.empty()) return shape;
  if (input_count == 1 || index == 0) return {1, 192};
  return {1, 256, 256, 3};
}

void FillTensor(mindspore::MSTensor *tensor) {
  void *data = tensor->MutableData();
  if (data == nullptr) {
    throw std::runtime_error("MutableData returned null for " + tensor->Name());
  }
  const auto bytes = tensor->DataSize();
  const auto dtype = tensor->DataType();
  if (dtype == mindspore::DataType::kNumberTypeFloat32) {
    std::fill_n(static_cast<float *>(data), bytes / sizeof(float), 0.0f);
  } else if (dtype == mindspore::DataType::kNumberTypeInt32) {
    std::fill_n(static_cast<int32_t *>(data), bytes / sizeof(int32_t), 0);
  } else if (dtype == mindspore::DataType::kNumberTypeInt64) {
    std::fill_n(static_cast<int64_t *>(data), bytes / sizeof(int64_t), 0);
  } else {
    std::memset(data, 0, bytes);
  }
}

void PrintTensor(const char *prefix, const mindspore::MSTensor &tensor, size_t index) {
  std::cout << prefix << "[" << index << "]"
            << " name=" << tensor.Name()
            << " dtype=" << DataTypeText(tensor.DataType())
            << " shape=" << ShapeText(tensor.Shape())
            << " elements=" << tensor.ElementNum()
            << " bytes=" << tensor.DataSize()
            << std::endl;
}

void PrintOutputSample(const mindspore::MSTensor &tensor) {
  auto data = tensor.Data();
  if (data == nullptr || data.get() == nullptr) {
    std::cout << "output sample unavailable" << std::endl;
    return;
  }
  std::cout << "output sample=";
  if (tensor.DataType() == mindspore::DataType::kNumberTypeFloat32) {
    const auto *ptr = static_cast<const float *>(data.get());
    const int64_t n = std::min<int64_t>(tensor.ElementNum(), 12);
    for (int64_t i = 0; i < n; ++i) std::cout << std::setprecision(6) << ptr[i] << " ";
  } else {
    const auto *ptr = static_cast<const uint8_t *>(data.get());
    const size_t n = std::min<size_t>(tensor.DataSize(), 12);
    for (size_t i = 0; i < n; ++i) std::cout << static_cast<int>(ptr[i]) << " ";
  }
  std::cout << std::endl;
}

int Smoke(const std::string &model_path) {
  std::ifstream file(model_path, std::ios::binary | std::ios::ate);
  if (!file.good()) {
    std::cerr << "model not found: " << model_path << std::endl;
    return 2;
  }
  std::cout << "model=" << model_path << std::endl;
  std::cout << "model_size=" << static_cast<long long>(file.tellg()) << std::endl;

  auto context = std::make_shared<mindspore::Context>();
  context->SetThreadNum(2);
  context->SetThreadAffinity(1);
  auto cpu = std::make_shared<mindspore::CPUDeviceInfo>();
  context->MutableDeviceInfo().push_back(cpu);

  mindspore::Model model;
  auto build_ret = model.Build(model_path, mindspore::kMindIR, context);
  std::cout << "build_status=" << build_ret << std::endl;
  if (build_ret != mindspore::kSuccess) return 3;

  auto inputs = model.GetInputs();
  std::cout << "input_count=" << inputs.size() << std::endl;
  for (size_t i = 0; i < inputs.size(); ++i) PrintTensor("input_before", inputs[i], i);
  if (inputs.empty()) return 4;

  std::vector<std::vector<int64_t>> resize_shapes;
  resize_shapes.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) resize_shapes.push_back(TargetShape(inputs[i], i, inputs.size()));
  auto resize_ret = model.Resize(inputs, resize_shapes);
  std::cout << "resize_status=" << resize_ret << std::endl;
  if (resize_ret != mindspore::kSuccess) return 5;

  inputs = model.GetInputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    PrintTensor("input_after", inputs[i], i);
    FillTensor(&inputs[i]);
  }

  auto outputs = model.GetOutputs();
  std::cout << "output_count_before=" << outputs.size() << std::endl;
  for (size_t i = 0; i < outputs.size(); ++i) PrintTensor("output_before", outputs[i], i);

  auto predict_ret = model.Predict(inputs, &outputs);
  std::cout << "predict_status=" << predict_ret << std::endl;
  if (predict_ret != mindspore::kSuccess) return 6;

  std::cout << "output_count_after=" << outputs.size() << std::endl;
  for (size_t i = 0; i < outputs.size(); ++i) {
    PrintTensor("output_after", outputs[i], i);
    PrintOutputSample(outputs[i]);
  }
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: mslite_smoke <model.ms>" << std::endl;
    return 1;
  }
  try {
    return Smoke(argv[1]);
  } catch (const std::exception &e) {
    std::cerr << "exception=" << e.what() << std::endl;
    return 10;
  }
}
