#ifndef AABHAS_RUST_EXPORTS_H
#define AABHAS_RUST_EXPORTS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    AABHAS_STATUS_OK = 0,
    AABHAS_STATUS_INVALID_ARGUMENT = -1,
    AABHAS_STATUS_ALREADY_STARTED = -2,
    AABHAS_STATUS_RAM_THROTTLED = -3,
    AABHAS_STATUS_SSD_RESERVE = -4,
    AABHAS_STATUS_HEALTH_CHECK = -5,
    AABHAS_STATUS_INTERNAL_PANIC = -6,
    AABHAS_STATUS_NOT_STARTED = -7
};

int32_t start_aabhas_engine(const char* model_path);
int32_t stop_aabhas_engine(void);

#ifdef __cplusplus
}
#endif

#endif /* AABHAS_RUST_EXPORTS_H */
