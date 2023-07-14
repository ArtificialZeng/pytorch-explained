//  Copyright © 2022 Apple Inc.

#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSGeneratorImpl.h>
#include <ATen/mps/MPSHooks.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/mps/MPSStream.h>
#include <c10/util/Logging.h>

namespace at {
namespace mps {

void MPSHooks::initMPS() const {
  C10_LOG_API_USAGE_ONCE("aten.init.mps");
  // TODO: initialize MPS devices and streams here
}

bool MPSHooks::hasMPS() const {
  return at::mps::is_available();
}

bool MPSHooks::isOnMacOS13orNewer(unsigned minor) const {
  switch (minor) {
    case 0:
      return is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_0_PLUS);
    case 1:
      return is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_1_PLUS);
    case 2:
      return is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_2_PLUS);
    case 3:
      return is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_3_PLUS);
    default:
      TORCH_WARN("Can't check whether running on 13.", minor, "+ returning one for 13.3+");
      return is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_3_PLUS);
  }
}

Allocator* MPSHooks::getMPSDeviceAllocator() const {
  return at::mps::GetMPSAllocator();
}

const Generator& MPSHooks::getDefaultMPSGenerator() const {
  return at::mps::detail::getDefaultMPSGenerator();
}

void MPSHooks::deviceSynchronize() const {
  at::mps::getDefaultMPSStream()->synchronize(SyncType::COMMIT_AND_WAIT);
}

void MPSHooks::commitStream() const {
  at::mps::getDefaultMPSStream()->synchronize(SyncType::COMMIT);
}

void* MPSHooks::getCommandBuffer() const {
  return at::mps::getDefaultMPSStream()->commandBuffer();
}

void* MPSHooks::getDispatchQueue() const {
  return at::mps::getDefaultMPSStream()->queue();
}

void MPSHooks::emptyCache() const {
  at::mps::getIMPSAllocator()->emptyCache();
}

size_t MPSHooks::getCurrentAllocatedMemory() const {
  return at::mps::getIMPSAllocator()->getCurrentAllocatedMemory();
}

size_t MPSHooks::getDriverAllocatedMemory() const {
  return at::mps::getIMPSAllocator()->getDriverAllocatedMemory();
}

void MPSHooks::setMemoryFraction(double ratio) const {
  at::mps::getIMPSAllocator()->setHighWatermarkRatio(ratio);
}

void MPSHooks::profilerStartTrace(const std::string& mode, bool waitUntilCompleted) const {
  at::mps::getMPSProfiler().StartTrace(mode, waitUntilCompleted);
}

void MPSHooks::profilerStopTrace() const {
  at::mps::getMPSProfiler().StopTrace();
}

using at::MPSHooksRegistry;
using at::RegistererMPSHooksRegistry;

REGISTER_MPS_HOOKS(MPSHooks);

} // namespace mps
} // namespace at
