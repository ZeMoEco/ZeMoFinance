#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <hilog/log.h>
#include <mindspore/context.h>
#include <mindspore/model.h>
#include <mindspore/status.h>
#include <mindspore/tensor.h>
#include <mindspore/types.h>
#include <napi/native_api.h>

#undef LOG_DOMAIN
#define LOG_DOMAIN 0x0D5B63
#undef LOG_TAG
#define LOG_TAG "ZemoMSLiteNative"

namespace {

OH_AI_ModelHandle g_model = nullptr;
std::string g_model_path;
std::mutex g_mutex;

std::string JsonEscape(const std::string &value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (char c : value) {
    switch (c) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += c;
        break;
    }
  }
  return out;
}

std::string MakeLoadJson(bool ok, int input_count, const std::string &message, const std::string &diag) {
  std::ostringstream oss;
  oss << "{\"ok\":" << (ok ? "true" : "false")
      << ",\"inputCount\":" << input_count
      << ",\"message\":\"" << JsonEscape(message)
      << "\",\"diag\":\"" << JsonEscape(diag) << "\"}";
  return oss.str();
}

napi_value MakeString(napi_env env, const std::string &value) {
  napi_value result = nullptr;
  napi_create_string_utf8(env, value.c_str(), value.size(), &result);
  return result;
}

std::string GetStringArg(napi_env env, napi_value value) {
  size_t len = 0;
  napi_get_value_string_utf8(env, value, nullptr, 0, &len);
  std::vector<char> buffer(len + 1, '\0');
  if (len > 0) napi_get_value_string_utf8(env, value, buffer.data(), buffer.size(), &len);
  std::string out(buffer.data(), len);
  return out;
}

void ThrowError(napi_env env, const std::string &message) {
  napi_throw_error(env, nullptr, message.c_str());
}

void DestroyModelLocked() {
  if (g_model != nullptr) {
    OH_AI_ModelDestroy(&g_model);
    g_model = nullptr;
  }
  g_model_path.clear();
}

std::string ShapeText(const int64_t *shape, size_t shape_num) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape_num; ++i) {
    if (i > 0) oss << ",";
    oss << shape[i];
  }
  oss << "]";
  return oss.str();
}

std::string TensorDiag(OH_AI_TensorHandle tensor) {
  if (tensor == nullptr) return "null";
  const char *name = OH_AI_TensorGetName(tensor);
  size_t shape_num = 0;
  const int64_t *shape = OH_AI_TensorGetShape(tensor, &shape_num);
  std::ostringstream oss;
  oss << (name == nullptr ? "input" : name)
      << ShapeText(shape, shape_num)
      << ",dtype=" << static_cast<int>(OH_AI_TensorGetDataType(tensor))
      << ",bytes=" << OH_AI_TensorGetDataSize(tensor);
  return oss.str();
}

std::string InputsDiagLocked() {
  if (g_model == nullptr) return "model=null";
  OH_AI_TensorHandleArray inputs = OH_AI_ModelGetInputs(g_model);
  std::ostringstream oss;
  oss << "getInputs=" << inputs.handle_num;
  if (inputs.handle_num > 0) oss << ":";
  const size_t count = std::min<size_t>(inputs.handle_num, 4);
  for (size_t i = 0; i < count; ++i) {
    if (i > 0) oss << "|";
    oss << TensorDiag(inputs.handle_list[i]);
  }
  return oss.str();
}

bool HasDynamicShape(OH_AI_TensorHandle tensor) {
  size_t shape_num = 0;
  const int64_t *shape = OH_AI_TensorGetShape(tensor, &shape_num);
  if (shape == nullptr || shape_num == 0) return true;
  for (size_t i = 0; i < shape_num; ++i) {
    if (shape[i] <= 0) return true;
  }
  return false;
}

int InputCountLocked() {
  if (g_model == nullptr) return 0;
  OH_AI_TensorHandleArray inputs = OH_AI_ModelGetInputs(g_model);
  return static_cast<int>(inputs.handle_num);
}

OH_AI_ContextHandle CreateCpuContext() {
  OH_AI_ContextHandle context = OH_AI_ContextCreate();
  if (context == nullptr) return nullptr;
  OH_AI_ContextSetThreadNum(context, 2);
  OH_AI_ContextSetThreadAffinityMode(context, 1);
  OH_AI_DeviceInfoHandle cpu_device_info = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
  if (cpu_device_info == nullptr) {
    OH_AI_ContextDestroy(&context);
    return nullptr;
  }
  OH_AI_ContextAddDeviceInfo(context, cpu_device_info);
  return context;
}

napi_value LoadModel(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1] = {nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  if (argc < 1) return MakeString(env, MakeLoadJson(false, 0, "missing modelPath", ""));
  const std::string model_path = GetStringArg(env, args[0]);
  if (model_path.empty()) return MakeString(env, MakeLoadJson(false, 0, "empty modelPath", ""));

  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_model != nullptr && g_model_path == model_path) {
    return MakeString(env, MakeLoadJson(true, InputCountLocked(), "loaded", InputsDiagLocked()));
  }

  DestroyModelLocked();
  OH_AI_ContextHandle context = CreateCpuContext();
  if (context == nullptr) return MakeString(env, MakeLoadJson(false, 0, "OH_AI_ContextCreate failed", ""));

  OH_AI_ModelHandle model = OH_AI_ModelCreate();
  if (model == nullptr) {
    OH_AI_ContextDestroy(&context);
    return MakeString(env, MakeLoadJson(false, 0, "OH_AI_ModelCreate failed", ""));
  }

  OH_AI_Status status = OH_AI_ModelBuildFromFile(model, model_path.c_str(), OH_AI_MODELTYPE_MINDIR, context);
  OH_AI_ContextDestroy(&context);
  if (status != OH_AI_STATUS_SUCCESS) {
    OH_AI_ModelDestroy(&model);
    std::ostringstream msg;
    msg << "OH_AI_ModelBuildFromFile failed, status=" << static_cast<int>(status);
    OH_LOG_ERROR(LOG_APP, "load failed: %{public}s", msg.str().c_str());
    return MakeString(env, MakeLoadJson(false, 0, msg.str(), model_path));
  }

  g_model = model;
  g_model_path = model_path;
  const int input_count = InputCountLocked();
  const std::string diag = InputsDiagLocked();
  OH_LOG_INFO(LOG_APP, "loaded model inputs=%{public}d diag=%{public}s", input_count, diag.c_str());
  return MakeString(env, MakeLoadJson(input_count > 0, input_count, input_count > 0 ? "loaded" : "model input empty", diag));
}

void FillInputIds(OH_AI_TensorHandle tensor, const int32_t *input_ids, size_t length) {
  const auto dtype = OH_AI_TensorGetDataType(tensor);
  const int64_t element_num = OH_AI_TensorGetElementNum(tensor);
  const size_t element_count = element_num <= 0 ? 0 : static_cast<size_t>(element_num);
  const size_t count = std::min<size_t>(length, element_count);
  void *data = OH_AI_TensorGetMutableData(tensor);
  if (data == nullptr) throw std::runtime_error("input_ids mutable data is null");
  if (dtype == OH_AI_DATATYPE_NUMBERTYPE_INT64) {
    auto *dst = static_cast<int64_t *>(data);
    for (size_t i = 0; i < count; ++i) dst[i] = static_cast<int64_t>(input_ids[i]);
    for (size_t i = count; i < element_count; ++i) dst[i] = 0;
  } else if (dtype == OH_AI_DATATYPE_NUMBERTYPE_INT32) {
    auto *dst = static_cast<int32_t *>(data);
    std::memcpy(dst, input_ids, count * sizeof(int32_t));
    if (element_count > count) std::memset(dst + count, 0, (element_count - count) * sizeof(int32_t));
  } else {
    const size_t bytes = OH_AI_TensorGetDataSize(tensor);
    std::memset(data, 0, bytes);
    std::memcpy(data, input_ids, std::min(bytes, count * sizeof(int32_t)));
  }
}

void FillFloatInput(OH_AI_TensorHandle tensor, const float *values, size_t length) {
  void *data = OH_AI_TensorGetMutableData(tensor);
  if (data == nullptr) throw std::runtime_error("pixel_values mutable data is null");
  const int64_t element_num = OH_AI_TensorGetElementNum(tensor);
  const size_t element_count = element_num <= 0 ? 0 : static_cast<size_t>(element_num);
  const size_t count = std::min<size_t>(length, element_count);
  const size_t bytes = OH_AI_TensorGetDataSize(tensor);
  std::memset(data, 0, bytes);
  if (OH_AI_TensorGetDataType(tensor) == OH_AI_DATATYPE_NUMBERTYPE_FLOAT32 && count > 0 && values != nullptr) {
    std::memcpy(data, values, std::min(bytes, count * sizeof(float)));
  }
}

float HalfToFloat(uint16_t value) {
  const uint32_t sign = (value & 0x8000u) << 16;
  uint32_t exp = (value >> 10) & 0x1fu;
  uint32_t frac = value & 0x03ffu;
  uint32_t out = 0;
  if (exp == 0) {
    if (frac == 0) {
      out = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((frac & 0x0400u) == 0) {
        frac <<= 1;
        --exp;
      }
      frac &= 0x03ffu;
      out = sign | (exp << 23) | (frac << 13);
    }
  } else if (exp == 31) {
    out = sign | 0x7f800000u | (frac << 13);
  } else {
    out = sign | ((exp + 127 - 15) << 23) | (frac << 13);
  }
  float result = 0.0f;
  std::memcpy(&result, &out, sizeof(float));
  return result;
}

std::vector<float> OutputToFloatVector(OH_AI_TensorHandle tensor) {
  const void *data = OH_AI_TensorGetData(tensor);
  const size_t bytes = OH_AI_TensorGetDataSize(tensor);
  if (data == nullptr || bytes == 0) return {};
  if (OH_AI_TensorGetDataType(tensor) == OH_AI_DATATYPE_NUMBERTYPE_FLOAT32) {
    const size_t count = bytes / sizeof(float);
    const auto *src = static_cast<const float *>(data);
    return std::vector<float>(src, src + count);
  }
  if (OH_AI_TensorGetDataType(tensor) == OH_AI_DATATYPE_NUMBERTYPE_FLOAT16) {
    const size_t count = bytes / sizeof(uint16_t);
    const auto *src = static_cast<const uint16_t *>(data);
    std::vector<float> out(count);
    for (size_t i = 0; i < count; ++i) out[i] = HalfToFloat(src[i]);
    return out;
  }
  return {};
}

napi_value FloatVectorToTypedArray(napi_env env, const std::vector<float> &values) {
  napi_value array_buffer = nullptr;
  void *data = nullptr;
  const size_t bytes = values.size() * sizeof(float);
  napi_create_arraybuffer(env, bytes, &data, &array_buffer);
  if (bytes > 0) std::memcpy(data, values.data(), bytes);
  napi_value typed_array = nullptr;
  napi_create_typedarray(env, napi_float32_array, values.size(), array_buffer, 0, &typed_array);
  return typed_array;
}

napi_value Predict(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3] = {nullptr, nullptr, nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  if (argc < 3) {
    ThrowError(env, "predict requires inputIds, pixelValues, imageSize");
    return nullptr;
  }

  napi_typedarray_type ids_type;
  size_t ids_len = 0;
  void *ids_data = nullptr;
  napi_value ids_ab = nullptr;
  size_t ids_offset = 0;
  napi_get_typedarray_info(env, args[0], &ids_type, &ids_len, &ids_data, &ids_ab, &ids_offset);
  if (ids_type != napi_int32_array || ids_data == nullptr || ids_len == 0) {
    ThrowError(env, "inputIds must be Int32Array");
    return nullptr;
  }

  napi_typedarray_type pixels_type;
  size_t pixels_len = 0;
  void *pixels_data = nullptr;
  napi_value pixels_ab = nullptr;
  size_t pixels_offset = 0;
  napi_get_typedarray_info(env, args[1], &pixels_type, &pixels_len, &pixels_data, &pixels_ab, &pixels_offset);
  if (pixels_type != napi_float32_array || (pixels_len > 0 && pixels_data == nullptr)) {
    ThrowError(env, "pixelValues must be Float32Array");
    return nullptr;
  }

  int32_t image_size = 0;
  napi_get_value_int32(env, args[2], &image_size);
  if (image_size <= 0) image_size = 256;

  try {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_model == nullptr) {
      ThrowError(env, "MiniMind-O native model not loaded");
      return nullptr;
    }
    OH_AI_TensorHandleArray inputs = OH_AI_ModelGetInputs(g_model);
    if (inputs.handle_num < 1) {
      ThrowError(env, "MiniMind-O native model input empty");
      return nullptr;
    }

    if (HasDynamicShape(inputs.handle_list[0]) || (inputs.handle_num >= 2 && HasDynamicShape(inputs.handle_list[1]))) {
      std::vector<OH_AI_ShapeInfo> shapes;
      shapes.resize(inputs.handle_num >= 2 ? 2 : 1);
      shapes[0].shape_num = 2;
      shapes[0].shape[0] = 1;
      shapes[0].shape[1] = static_cast<int64_t>(ids_len);
      if (inputs.handle_num >= 2) {
        shapes[1].shape_num = 4;
        shapes[1].shape[0] = 1;
        shapes[1].shape[1] = image_size;
        shapes[1].shape[2] = image_size;
        shapes[1].shape[3] = 3;
      }
      OH_AI_Status resize_status = OH_AI_ModelResize(g_model, inputs, shapes.data(), shapes.size());
      if (resize_status != OH_AI_STATUS_SUCCESS) {
        OH_LOG_WARN(LOG_APP, "resize returned %{public}d, continue with current input shapes", static_cast<int>(resize_status));
      }
      inputs = OH_AI_ModelGetInputs(g_model);
    }
    FillInputIds(inputs.handle_list[0], static_cast<const int32_t *>(ids_data), ids_len);
    if (inputs.handle_num >= 2) {
      FillFloatInput(inputs.handle_list[1], static_cast<const float *>(pixels_data), pixels_len);
    }

    OH_AI_TensorHandleArray outputs = {};
    OH_AI_Status predict_status = OH_AI_ModelPredict(g_model, inputs, &outputs, nullptr, nullptr);
    if (predict_status != OH_AI_STATUS_SUCCESS || outputs.handle_num == 0) {
      std::ostringstream msg;
      msg << "OH_AI_ModelPredict failed, status=" << static_cast<int>(predict_status)
          << ", outputs=" << outputs.handle_num
          << ", ids_len=" << ids_len
          << ", pixels_len=" << pixels_len
          << ", inputs=" << InputsDiagLocked();
      OH_LOG_WARN(LOG_APP, "predict unavailable, fallback to empty logits: %{public}s", msg.str().c_str());
      return FloatVectorToTypedArray(env, std::vector<float>());
    }
    const std::vector<float> logits = OutputToFloatVector(outputs.handle_list[0]);
    if (logits.empty()) {
      ThrowError(env, "MiniMind-O native output is empty or unsupported dtype");
      return nullptr;
    }
    return FloatVectorToTypedArray(env, logits);
  } catch (const std::exception &e) {
    ThrowError(env, e.what());
    return nullptr;
  }
}

napi_value InputCount(napi_env env, napi_callback_info) {
  std::lock_guard<std::mutex> lock(g_mutex);
  napi_value value = nullptr;
  napi_create_int32(env, InputCountLocked(), &value);
  return value;
}

napi_value InputDiag(napi_env env, napi_callback_info) {
  std::lock_guard<std::mutex> lock(g_mutex);
  return MakeString(env, InputsDiagLocked());
}

napi_value Reset(napi_env env, napi_callback_info) {
  std::lock_guard<std::mutex> lock(g_mutex);
  DestroyModelLocked();
  napi_value undefined = nullptr;
  napi_get_undefined(env, &undefined);
  return undefined;
}

napi_value Init(napi_env env, napi_value exports) {
  napi_property_descriptor desc[] = {
    {"loadModel", nullptr, LoadModel, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"predict", nullptr, Predict, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"inputCount", nullptr, InputCount, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"inputDiag", nullptr, InputDiag, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"reset", nullptr, Reset, nullptr, nullptr, nullptr, napi_default, nullptr},
  };
  napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
  return exports;
}

}  // namespace

static napi_module g_zemo_mslite_native_module = {
  1,
  0,
  nullptr,
  Init,
  "zemo_mslite_native",
  nullptr,
  {0},
};

extern "C" __attribute__((constructor)) void RegisterZemoMindSporeNativeModule() {
  napi_module_register(&g_zemo_mslite_native_module);
}
