#include "scheduler/layer_scheduler.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

int main() {
    const auto path = std::filesystem::temp_directory_path() / "aabhas-layer-scheduler-test.bin";
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    assert(output.is_open());
    const std::vector<char> payload(12, 'x');
    output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    output.close();

    using aabhas::scheduler::LayerScheduler;
    using aabhas::scheduler::SchedulerConfig;
    using aabhas::tensor_ops::LayerDescriptor;
    LayerScheduler scheduler;
    const bool opened = scheduler.open(
        path.string().c_str(),
        std::vector<LayerDescriptor>{{0, 4}, {4, 4}, {8, 4}},
        SchedulerConfig{4, 4, 1});
    assert(opened);
    assert(scheduler.pending_layers() == 3);
    assert(scheduler.next_layer());
    assert(scheduler.has_current());
    assert(scheduler.current().size == 4);
    assert(!scheduler.next_layer());
    assert(scheduler.release());
    assert(scheduler.next_layer());
    assert(scheduler.current().size == 4);
    assert(scheduler.release());
    assert(scheduler.next_layer());
    assert(scheduler.release());
    assert(!scheduler.next_layer());

    scheduler.close();
    std::filesystem::remove(path);
    return 0;
}
