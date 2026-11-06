#include "hardware/hw_identifier.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#include <winreg.h>
#else
#include <fcntl.h>
#include <limits>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/utsname.h>
#include <unistd.h>
#endif

#if defined(__linux__)
#include <sys/sysmacros.h>
#endif

namespace aabhas::hardware {
namespace {

std::string trim_ascii(const char* value, std::size_t length) {
    std::size_t begin = 0;
    while (begin < length && (value[begin] == ' ' || value[begin] == '\t' ||
                              value[begin] == '\r' || value[begin] == '\n')) {
        ++begin;
    }
    std::size_t end = length;
    while (end > begin && (value[end - 1] == ' ' || value[end - 1] == '\t' ||
                           value[end - 1] == '\r' || value[end - 1] == '\n')) {
        --end;
    }
    return std::string(value + begin, end - begin);
}

#if defined(_WIN32)
std::string read_windows_machine_guid() {
    HKEY key = nullptr;
    const LONG opened = ::RegOpenKeyExA(
        HKEY_LOCAL_MACHINE, "SOFTWARE\\Microsoft\\Cryptography", 0,
        KEY_READ | KEY_WOW64_64KEY, &key);
    if (opened != ERROR_SUCCESS) return {};

    char value[256]{};
    DWORD value_size = static_cast<DWORD>(sizeof(value));
    DWORD value_type = 0;
    const LONG queried = ::RegQueryValueExA(
        key, "MachineGuid", nullptr, &value_type,
        reinterpret_cast<LPBYTE>(value), &value_size);
    (void)::RegCloseKey(key);
    if (queried != ERROR_SUCCESS ||
        (value_type != REG_SZ && value_type != REG_EXPAND_SZ) || value_size == 0) {
        return {};
    }
    const std::size_t length = value_size > 0 && value[value_size - 1] == '\0'
        ? static_cast<std::size_t>(value_size - 1)
        : static_cast<std::size_t>(value_size);
    return trim_ascii(value, length);
}
#else
std::string read_posix_identity_file(const char* path) {
    int flags = O_RDONLY;
#if defined(O_CLOEXEC)
    flags |= O_CLOEXEC;
#endif
    const int fd = ::open(path, flags);
    if (fd < 0) return {};
    char value[256]{};
    const ssize_t bytes = ::read(fd, value, sizeof(value) - 1);
    (void)::close(fd);
    if (bytes <= 0) return {};
    return trim_ascii(value, static_cast<std::size_t>(bytes));
}
#endif

#if !defined(_WIN32)
std::uint64_t positive_sysconf(int name) noexcept {
    const long value = ::sysconf(name);
    return value > 0 ? static_cast<std::uint64_t>(value) : 0;
}

std::uint64_t multiply_saturated(std::uint64_t lhs, std::uint64_t rhs) noexcept {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return lhs * rhs;
}

#if defined(__linux__)
DriveType rotational_device_type(const std::string& device) {
    const std::size_t block_marker = device.find("/block/");
    if (block_marker == std::string::npos) return DriveType::Unknown;
    const std::size_t name_begin = block_marker + 7;
    const std::size_t name_end = device.find('/', name_begin);
    const std::string name = device.substr(
        name_begin, name_end == std::string::npos ? std::string::npos : name_end - name_begin);
    if (name.empty()) return DriveType::Unknown;

    char rotational_path[256]{};
    const int path_size = ::snprintf(rotational_path, sizeof(rotational_path),
                                     "/sys/block/%s/queue/rotational", name.c_str());
    if (path_size <= 0 || static_cast<std::size_t>(path_size) >= sizeof(rotational_path)) {
        return DriveType::Unknown;
    }
#if defined(O_CLOEXEC)
    const int fd = ::open(rotational_path, O_RDONLY | O_CLOEXEC);
#else
    const int fd = ::open(rotational_path, O_RDONLY);
#endif
    if (fd < 0) return DriveType::Unknown;
    char value = '\0';
    const ssize_t bytes = ::read(fd, &value, 1);
    (void)::close(fd);
    if (bytes != 1) return DriveType::Unknown;
    return value == '0' ? DriveType::SSD : DriveType::HDD;
}

DriveType linux_device_type(const struct stat& metadata, const char* path) {
    struct stat device_metadata = metadata;
    if (!S_ISBLK(metadata.st_mode)) {
        device_metadata.st_rdev = metadata.st_dev;
    }
    const unsigned int major_number = ::major(device_metadata.st_rdev);
    const unsigned int minor_number = ::minor(device_metadata.st_rdev);
    char sysfs_link[128]{};
    const int link_size = ::snprintf(sysfs_link, sizeof(sysfs_link),
                                     "/sys/dev/block/%u:%u", major_number, minor_number);
    if (link_size > 0 && static_cast<std::size_t>(link_size) < sizeof(sysfs_link)) {
        char target[512]{};
        const ssize_t target_size = ::readlink(sysfs_link, target, sizeof(target) - 1);
        if (target_size > 0) {
            target[target_size] = '\0';
            const std::string device(target);
            if (device.find("nvme") != std::string::npos) return DriveType::NVMe;
            const DriveType rotational_type = rotational_device_type(device);
            if (rotational_type != DriveType::Unknown) return rotational_type;
        }
    }
    if (path != nullptr && std::strncmp(path, "/dev/nvme", 9) == 0) {
        return DriveType::NVMe;
    }
    return DriveType::Unknown;
}
#else
DriveType linux_device_type(const struct stat&, const char*) {
    return DriveType::Unknown;
}
#endif

#endif // !defined(_WIN32)

} // namespace

HostSpecifications HardwareIdentifier::host() const {
    HostSpecifications result;
#if defined(_WIN32)
    SYSTEM_INFO system_info{};
    ::GetSystemInfo(&system_info);
    result.system = "Windows";
    result.logical_cpu_count = static_cast<std::uint64_t>(system_info.dwNumberOfProcessors);
    result.page_size = static_cast<std::uint64_t>(system_info.dwPageSize);

    MEMORYSTATUSEX memory_status{};
    memory_status.dwLength = sizeof(memory_status);
    if (::GlobalMemoryStatusEx(&memory_status) != FALSE) {
        result.physical_memory_bytes = memory_status.ullTotalPhys;
        result.available_memory_bytes = memory_status.ullAvailPhys;
    }
#else
    struct utsname system_info {};
    if (::uname(&system_info) == 0) {
        result.system = system_info.sysname;
        result.release = system_info.release;
        result.machine = system_info.machine;
    }
#if defined(_SC_NPROCESSORS_ONLN)
    result.logical_cpu_count = positive_sysconf(_SC_NPROCESSORS_ONLN);
#endif
#if defined(_SC_PAGESIZE)
    result.page_size = positive_sysconf(_SC_PAGESIZE);
#elif defined(_SC_PAGE_SIZE)
    result.page_size = positive_sysconf(_SC_PAGE_SIZE);
#endif
#if defined(_SC_PHYS_PAGES)
    result.physical_memory_bytes = multiply_saturated(
        positive_sysconf(_SC_PHYS_PAGES), result.page_size);
#endif
#if defined(_SC_AVPHYS_PAGES)
    result.available_memory_bytes = multiply_saturated(
        positive_sysconf(_SC_AVPHYS_PAGES), result.page_size);
#endif
#endif
    return result;
}

DriveSpecifications HardwareIdentifier::drive(const char* path) const {
    DriveSpecifications result;
    if (path == nullptr) {
        errno = EINVAL;
        return result;
    }
    result.path = path;
#if defined(_WIN32)
    const DWORD attributes = ::GetFileAttributesA(path);
    if (attributes == INVALID_FILE_ATTRIBUTES) return result;
    result.regular_file = (attributes & FILE_ATTRIBUTE_DIRECTORY) == 0;
    ULARGE_INTEGER free_bytes{};
    ULARGE_INTEGER total_bytes{};
    ULARGE_INTEGER total_free_bytes{};
    if (::GetDiskFreeSpaceExA(path, &free_bytes, &total_bytes, &total_free_bytes) != FALSE) {
        result.capacity_bytes = total_bytes.QuadPart;
        result.available_bytes = free_bytes.QuadPart;
    }
    return result;
#else
    struct stat metadata {};
    if (::stat(path, &metadata) != 0) return result;
    result.regular_file = S_ISREG(metadata.st_mode);
    result.type = linux_device_type(metadata, path);

    struct statvfs filesystem_info {};
    if (::statvfs(path, &filesystem_info) == 0) {
        const std::uint64_t block_size = filesystem_info.f_frsize != 0
            ? static_cast<std::uint64_t>(filesystem_info.f_frsize)
            : static_cast<std::uint64_t>(filesystem_info.f_bsize);
        result.capacity_bytes = multiply_saturated(
            static_cast<std::uint64_t>(filesystem_info.f_blocks), block_size);
        result.available_bytes = multiply_saturated(
            static_cast<std::uint64_t>(filesystem_info.f_bavail), block_size);
    }
    return result;
#endif
}

std::string HardwareIdentifier::system_uuid() const {
#if defined(_WIN32)
    // MachineGuid is the stable native Windows host identity available without
    // introducing a COM/WMI runtime dependency. It is used only as a fallback
    // host binding when a firmware UUID API is not exposed to this process.
    return read_windows_machine_guid();
#else
    std::string uuid = read_posix_identity_file("/sys/class/dmi/id/product_uuid");
    if (!uuid.empty()) return uuid;
    uuid = read_posix_identity_file("/sys/devices/virtual/dmi/id/product_uuid");
    if (!uuid.empty()) return uuid;
    // machine-id is a stable host identity fallback for non-DMI environments.
    return read_posix_identity_file("/etc/machine-id");
#endif
}

const char* HardwareIdentifier::drive_type_name(DriveType type) noexcept {
    switch (type) {
    case DriveType::NVMe: return "NVMe";
    case DriveType::SSD: return "SSD";
    case DriveType::HDD: return "HDD";
    case DriveType::Other: return "Other";
    case DriveType::Unknown: return "Unknown";
    }
    return "Unknown";
}

} // namespace aabhas::hardware
