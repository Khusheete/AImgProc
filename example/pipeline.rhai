

fn sobel_x(scale) {
    [scale, 0.0, -scale, 2.0 * scale, 0.0, -2.0 * scale, scale, 0.0, -scale]
}


fn sobel_y(scale) {
    [scale, 2.0 * scale, scale, 0.0, 0.0, 0.0, -scale, -2.0 * scale, -scale]
}


fn gauss(size, sigma) {
    let kernel = [];
    let sigma2 = sigma * sigma;

    for i in 0..(size * size) {
        let x = (i % size) - (size / 2) + 0.0;
        let y = (i / size) - (size / 2) + 0.0;

        let tmp1 = -(x * x + y * y) / (2.0 * sigma2);
        let tmp2 = (PI() * 2.0 * sigma2);

        kernel.push(tmp1.exp() / tmp2);
    }

    kernel
}


fn init() {
    ocl.create_float_buffer("gauss", gauss(5, 1.0));
    ocl.create_float_buffer("sobel_x", sobel_x(0.5));
    ocl.create_float_buffer("sobel_y", sobel_y(0.5));

    ocl.create_dynimage("buffer");
}


fn run() {
    ocl.call_kernel("apply_kernel", [input , output, gauss  , 5, 5]);
    ocl.call_kernel("apply_kernel", [output, input , sobel_x, 3, 3]);
    ocl.call_kernel("apply_kernel", [output, buffer, sobel_y, 3, 3]);
    ocl.call_kernel("combine_sobel", [input, buffer, output]);

    // ocl.call_kernel("dim", [input, output, 3]);
}