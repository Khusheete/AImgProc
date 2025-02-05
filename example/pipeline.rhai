

fn sobel_x(scale) {
    [scale, 0.0, -scale, 2.0 * scale, 0.0, -2.0 * scale, scale, 0.0, -scale]
}


fn sobel_y(scale) {
    [scale, 2.0 * scale, scale, 0.0, 0.0, 0.0, -scale, -2.0 * scale, -scale]
}


fn laplacian() {
    [0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0]
}


fn laplacian_diag() {
    [-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0]
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
    if config?.gauss != () {
        ocl.create_float_buffer("gauss", gauss(5, 1.0));
    }

    switch config?.mode {
        "sobel" => {
            ocl.create_float_buffer("sobel_x", sobel_x(1.0));
            ocl.create_float_buffer("sobel_y", sobel_y(1.0));
        }
        "laplacian" => {
            ocl.create_float_buffer("laplacian", laplacian());
        }
        "laplacian_diag" => {
            ocl.create_float_buffer("laplacian", laplacian_diag());
        }
        _ => {
            print("Provide operating mode (mode=sobel|laplacian|laplacian_diag)");
        }
    }

    ocl.create_dynimage("buffer1");
    ocl.create_dynimage("buffer2");
}


fn run() {
    let a = input;
    let b = buffer1;
    let c = buffer2;

    if config?.gauss != () {
        ocl.call_kernel("apply_kernel", [input, buffer1, gauss, 5, 5]);
        a = buffer1;
        b = input;
    }

    switch config?.mode {
        "sobel" => {
            ocl.call_kernel("apply_kernel",  [a, b, sobel_x, 3, 3]);
            ocl.call_kernel("apply_kernel",  [a, c, sobel_y, 3, 3]);
            ocl.call_kernel("combine_sobel", [b, c, output, 1]);
        }
        "laplacian" => {
            ocl.call_kernel("apply_kernel", [a, output, laplacian, 3, 3]);
        }
        "laplacian_diag" => {
            ocl.call_kernel("apply_kernel", [a, output, laplacian, 3, 3]);
        }
    }
}
