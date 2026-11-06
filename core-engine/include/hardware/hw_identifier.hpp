#ifndef AABHAS_HARDWARE_HW_IDENTIFIER_HPP
#define AABHAS_HARDWARE_HW_IDENTIFIER_HPP

#include <cstdint>
#include <string>

namespace aabhas::hardware {

struct HostSpecifications {
    std::string system;
    std::string release;
    std::string machine;
    std::uint64_t logical_cpu_count{0};
    std::uint64_t page_size{0};
    std::uint64_t physical_memory_bytes{0};
    std::uint64_t available_memory_bytes{0};
};

enum class DriveType { Unknown, NVMe, SSD, HDD, Other };

struct DriveSpecifications {
    std::string path;
    DriveType type{DriveType::Unknown};
    std::uint64_t capacity_bytes{0};
    std::uint64_t available_bytes{0};
    bool regular_file{false};
};

/** Reads host and backing-store facts using POSIX system interfaces. */
class HardwareIdentifier final {
public:
    HostSpecifications host() const;
    DriveSpecifications drive(const char* path) const;
    /** Return a stable host UUID, or an empty string when unavailable. */
    std::string system_uuid() const;
    static const char* drive_type_name(DriveType type) noexcept;
};

} // namespace aabhas::hardware

#endif // AABHAS_HARDWARE_HW_IDENTIFIER_HPP
