// Image data struct
typedef struct {
    uchar r, g, b;
} Color;


typedef struct {
    uint2 size;
    __global Color* px;
} Image;


// Image manipulation code
Color get_pixel(const uint2 pos, const Image img) {
    if (pos.x >= 0 && pos.x < img.size.x && pos.y >= 0 && pos.y < img.size.y) {
        return img.px[pos.x + pos.y * img.size.x];
    } else {
        const Color BLACK = {};
        return BLACK;
    }
}


void set_pixel(const uint2 pos, const Color c, Image img) {
    if (pos.x >= 0 && pos.x < img.size.x && pos.y >= 0 && pos.y < img.size.y) {
        img.px[pos.x + pos.y * img.size.x] = c;
    }
}


// kernels and user code
#define INIT() \
const uint2 pos = {get_global_id(0), get_global_id(1)}; \
const uint2 size = {img_w, img_h};                      \
if (pos.x >= size.x || pos.y >= size.y) {               \
    return;                                             \
}

#define IMAGE(name, data) Image name = {size, data}


float constrain(float val, float min, float max) {
    if (val < min) return min;
    else if (val > max) return max;
    else return val;
}


__kernel void dim(__global Color* in_colors, __global Color* out_colors,
    const long factor, const uint img_w, const uint img_h) 
{
    INIT();
    const IMAGE(in, in_colors);
    IMAGE(out, out_colors);

    Color px = get_pixel(pos, in);
    px.r /= factor;
    px.g /= factor;
    px.b /= factor;

    set_pixel(pos, px, out);
}


typedef struct {
    ulong2 size;
    __constant double* data;
} Kernel;


// Returns the kernel value at `(x, y)`
float get_ker_val(uint2 pos, const Kernel ker)
{
    pos.x += ker.size.x / 2;
    pos.y += ker.size.y / 2;
    if (pos.x < 0 || pos.x >= ker.size.x || pos.y < 0 || pos.y >= ker.size.y) {
        return 0;
    } else {
        return ker.data[pos.x + pos.y * ker.size.x];
    }
}


__kernel void apply_kernel(__global Color* in_colors, __global Color* out_colors,
    __constant double* ker_data, const ulong ker_w, const ulong ker_h,
    const uint img_w, const uint img_h) 
{
    INIT();
    const IMAGE(in, in_colors);
    IMAGE(out, out_colors);

    // setup kernel struct
    const ulong2 ksize = {ker_w, ker_h};
    const Kernel ker = {ksize, ker_data};

    // calculate new color
    float r = 0;
    float g = 0;
    float b = 0;

    for (int i = 0; i < ker_w; i++) {
        for (int j = 0; j < ker_h; j++) {
            uint2 d = {i - ker_w / 2, j - ker_h / 2};

            Color px = get_pixel(pos + d, in);
            float k_val = get_ker_val(d, ker);

            r += k_val * px.r;
            g += k_val * px.g;
            b += k_val * px.b;
        }
    }
    
    r = constrain(r, 0, 255);
    g = constrain(g, 0, 255);
    b = constrain(b, 0, 255);

    // set pixel color
    Color px = {r, g, b};
    set_pixel(pos, px, out);
}


__kernel void combine_sobel(__global Color* img0_colors, __global Color* img1_colors,
    __global Color* out_colors, const uint img_w, const uint img_h)
{
    INIT();
    const IMAGE(img0, img0_colors);
    const IMAGE(img1, img1_colors);
    IMAGE(out, out_colors);

    // combine colors
    Color px0 = get_pixel(pos, img0);
    Color px1 = get_pixel(pos, img1);

    int r = sqrt((double)(px0.r * px0.r + px1.r * px1.r));
    int g = sqrt((double)(px0.g * px0.g + px1.g * px1.g));
    int b = sqrt((double)(px0.b * px0.b + px1.b * px1.b));

    // int r = (atan2pi((double) px1.r, (double) px0.r) + 0.5f) * 255.0f;
    // int g = (atan2pi((double) px1.g, (double) px0.g) + 0.5f) * 255.0f;
    // int b = (atan2pi((double) px1.b, (double) px0.b) + 0.5f) * 255.0f;

    r = constrain(r, 0, 255);
    g = constrain(g, 0, 255);
    b = constrain(b, 0, 255);

    // set pixel
    Color px = {r, g, b};
    set_pixel(pos, px, out);
}