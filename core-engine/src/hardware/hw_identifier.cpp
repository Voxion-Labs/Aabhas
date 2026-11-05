#include "hardware/hw_identifier.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/utsname.h>
#include <unistd.h>

#if defined(__linux__)
#include <sys/sysmacros.h>
#endif

namespace aabhas::hardware {
namespace {

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

} // namespace

HostSpecifications HardwareIdentifier::host() const {
    HostSpecifications result;
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
    return result;
}

DriveSpecifications HardwareIdentifier::drive(const char* path) const {
    DriveSpecifications result;
    if (path == nullptr) {
        errno = EINVAL;
        return result;
    }
    result.path = path;
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
