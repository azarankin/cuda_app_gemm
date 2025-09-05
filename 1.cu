#include <nvToolsExt.h>
#include <cuda_runtime.h>
#include <string>
#include <functional>

// טיפוסים ייעודיים למידע
struct color {
    uint32_t value;
    explicit color(uint32_t c) : value(c) {}
};

struct name {
    std::string value;
    explicit name(const std::string& s) : value(s) {}
};

// מחלקת ביניים
class range_builder {
    std::string label = "Unnamed";
    uint32_t col = 0xFF00FF00; // ברירת מחדל: ירוק
public:
    range_builder& operator<<(const name& n) {
        label = n.value;
        return *this;
    }
    range_builder& operator<<(const color& c) {
        col = c.value;
        return *this;
    }

    void operator<<(const std::function<void()>& fn) {
        nvtxEventAttributes_t attr = {};
        attr.version = NVTX_VERSION;
        attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
        attr.message.ascii = label.c_str();
        attr.colorType = NVTX_COLOR_ARGB;
        attr.color = col;

        nvtxRangePushEx(&attr);
        fn();
        cudaDeviceSynchronize();
        nvtxRangePop();
    }
};

// מחלקה ראשית
class profile_range {
public:
    range_builder operator<<(const name& n) {
        range_builder rb;
        rb << n;
        return rb;
    }
    range_builder operator<<(const color& c) {
        range_builder rb;
        rb << c;
        return rb;
    }
};
