/*
MIT License

Copyright (c) 2022 Siandfrance

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

extern crate ocl;
extern crate image;
extern crate clap;
extern crate rhai;

mod formats;
mod compute;

use clap::Parser;

use compute::CInstance;

use image::RgbImage;
use image::io::Reader as ImageReader;

use std::path::Path;


pub const RED:   &str = "\x1b[38;2;255;0;0m";
pub const GREEN: &str = "\x1b[38;2;0;255;0m";
pub const CLEAR: &str = "\x1b[m";


/// An image processing program for use in AI image recognition
#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Source data
    #[clap(value_parser)]
    src: Option<String>,
    /// Opencl program to be used
    #[clap(value_parser)]
    program: Option<String>,
    /// Rhai script pipeline
    #[clap(value_parser)]
    pipeline: Option<String>,

    #[clap(value_parser)]
    /// The maximum width of the images to process
    width: Option<usize>,
    #[clap(value_parser)]
    /// The maximum height of the images to process
    height: Option<usize>,

    #[clap(short, long, value_parser, default_value_t = String::from("out"))]
    /// Output file or directory
    output: String,

    /// List all available platforms and devices
    #[clap(short = 'l', long, action)]
    list_platform: bool,

    /// rhai script configuration
    #[clap(short, long, value_parser)]
    config: Option<String>,

    #[clap(short, long, action)]
    verbose: bool
}


// TODO: select device from command line (with default)


fn main() {
    let args = Args::parse();

    if args.list_platform {
        list_platform(args.verbose);
    } else {

        let src = match args.src {
            None => {
                eprintln!("{}Provide source image or directory to process.{}", RED, CLEAR);
                eprintln!("To print help use --help.");
                return;
            },
            Some(s) => s
        };

        let program = match args.program {
            None => {
                eprintln!("{}Provide the opencl program.{}", RED, CLEAR);
                eprintln!("To print help use --help.");
                return;
            },
            Some(s) => s
        };

        let pipeline = match args.pipeline {
            None => {
                eprintln!("{}Provide a pipepline to follow.{}", RED, CLEAR);
                eprintln!("To print help use --help.");
                return;
            },
            Some(s) => s
        };


        let size = match (args.width, args.height) {
            (Some(w), Some(h)) => (w, h),
            _ => {
                eprintln!("{}Provide the maximum image dimentions.{}", RED, CLEAR);
                eprintln!("To print help use --help.");
                return;
            }
        };


        let config = match args.config {
            Some(c) => c,
            None => String::from("{}")
        };

        let mut compute = CInstance::init(args.verbose, program, pipeline, config, size);

        use std::fs::metadata;

        let src_meta = metadata(format!("{}", &src)).expect(format!("File `{}` does not exist", src).as_str());

        if src_meta.is_dir() {
            process_dir(&mut compute, Path::new(&src), Path::new(&args.output));
        } else if src_meta.is_file() {
            process_file(&mut compute, Path::new(&src), Path::new(&args.output));
        }
    }
}


/// Applies the compute pipeline to the input file, saving it to out_file
fn process_file(compute: &mut CInstance, in_file: &Path, out_file: &Path) {
    let img = ImageReader::open(in_file)
        .expect(format!("Could not read file `{}`", in_file.to_str().unwrap()).as_str()).decode()
        .expect(format!("Could not read image at `{}`", in_file.to_str().unwrap()).as_str());
    let image: RgbImage = img.into_rgb8();

    let out = compute.compute(&image);
    out.save(out_file)
        .expect(format!("Could not save image to `{}`", out_file.to_str().unwrap()).as_str());
}


fn process_dir(compute: &mut CInstance, in_dir: &Path, out_dir: &Path) {
    use std::fs;

    let file_count = fs::read_dir(in_dir)
        .expect(format!("Could not read files in `{}`", in_dir.to_str().unwrap()).as_str())
        .count();
    
    let mut i = 0;

    println!("<----------------------------------------> 0.00%");

    for file in fs::read_dir(in_dir).unwrap() {
        match file {
            Ok(file) => {
                if file.file_type().unwrap().is_file() {
                    let mut in_file = in_dir.to_path_buf();
                    in_file.push(file.file_name());

                    let mut out_file = out_dir.to_path_buf();
                    out_file.push(file.file_name());

                    process_file(compute, in_file.as_path(), out_file.as_path());
                }
            }
            _ => {}
        }

        i += 1;
        let progress_percent = (i as f32 / file_count as f32) * 100.0;
        let progress = ((i as f32 / file_count as f32) * 40.0) as i32;
        print!("\x1b[A\r<");
        for _ in 0..progress {
            print!("=");
        }
        for _ in progress..40 {
            print!("-");
        }
        println!("> {:.2}%", progress_percent);
    }
}


/// Lists all available platforms in a comprehensible way
fn list_platform(verbose: bool) {
    use formats::*;

    use ocl::{Platform, Device, enums::{DeviceInfo, DeviceInfoResult as DIR, DeviceMemCacheType, DeviceLocalMemType}};
    use ocl::flags::{DEVICE_TYPE_CPU, DEVICE_TYPE_GPU, DEVICE_TYPE_ACCELERATOR,
                    DEVICE_TYPE_CUSTOM, DEVICE_TYPE_DEFAULT};

    let platforms = Platform::list();

    if platforms.len() == 0 {
        println!("{}No platforms found on this machine. \nTry to install opencl packages.{}", RED, CLEAR);
    }

    for p in platforms {
        // println!("platform: {}{:?}{}", GREEN, p.as_core(), CLEAR);
        if let Ok(name) = p.name() {
            println!("name: {}", name);
        } else {
            println!("  {}Could not get platform name.{}", RED, CLEAR);
        }
        if let Ok(vendor) = p.vendor() {
            println!("  vendor: {}", vendor);
        }
        if let Ok(version) = p.version() {
            println!("  version: {}", version);
        }

        if let Ok(devices) = Device::list(p, None) {
            if devices.len() == 0 {
                println!("    {}No devices found on this platform.{}", RED, CLEAR);
            }

            for d in devices {
                println!();
                if let Ok(name) = d.name() {
                    println!("  device name: {}", name);
                } else {
                    println!("  {}Could not get device name.{}", RED, CLEAR);
                }
                if let Ok(DIR::Type(tpe)) = d.info(DeviceInfo::Type) {
                    print!("  type: ");
                    if tpe.contains(DEVICE_TYPE_DEFAULT) {
                        print!("default ");
                    }
                    if tpe.contains(DEVICE_TYPE_CPU) {
                        print!("CPU ");
                    }
                    if tpe.contains(DEVICE_TYPE_GPU) {
                        print!("GPU ");
                    }
                    if tpe.contains(DEVICE_TYPE_ACCELERATOR) {
                        print!("accelerator ");
                    }
                    if tpe.contains(DEVICE_TYPE_CUSTOM) {
                        print!("custom ")
                    }
                    println!();
                }
                if let Ok(vendor) = d.vendor() {
                    println!("    vendor: {}", vendor);
                }
                if let Ok(version) = d.version() {
                    println!("    opencl version: {}", version);
                }
                if let Ok(DIR::DriverVersion(version)) = d.info(DeviceInfo::DriverVersion) {
                    println!("    driver version: {}", version);
                }
                if let Ok(available) = d.is_available() {
                    println!("    available: {}", format_bool(available));
                }

                
                if verbose {

                    // general information about the device
                    if let Ok(DIR::MaxComputeUnits(mx)) = d.info(DeviceInfo::MaxComputeUnits) {
                        println!("    max compute units: {}", mx);
                    }
                    if let Ok(DIR::MaxWorkItemDimensions(mx)) = d.info(DeviceInfo::MaxWorkItemDimensions) {
                        println!("    max work item dimensions: {}", mx);
                    }
                    if let Ok(max_wg_size) = d.max_wg_size() {
                        println!("    max workgroup size: {}", max_wg_size);
                    }
                    if let Ok(DIR::MaxClockFrequency(mx)) = d.info(DeviceInfo::MaxClockFrequency) {
                        println!("    max clock frequency: {}", format_freq(mx as f32));
                    }
                    if let Ok(DIR::MaxMemAllocSize(mx)) = d.info(DeviceInfo::MaxMemAllocSize) {
                        println!("    max memory alloc size: {}", format_mem(mx));
                    }
                    if let Ok(DIR::MaxParameterSize(mx)) = d.info(DeviceInfo::MaxParameterSize) {
                        println!("    max parameter size: {}", mx);
                    }
                    if let Ok(DIR::MaxSamplers(mx)) = d.info(DeviceInfo::MaxSamplers) {
                        println!("    max samplers: {}", mx);
                    }
                    

                    // images
                    if let Ok(DIR::ImageSupport(b)) = d.info(DeviceInfo::ImageSupport) {
                        println!("    image support: {}", format_bool(b));
                    }
                    if let (Ok(DIR::Image2dMaxWidth(w)), Ok(DIR::Image2dMaxHeight(h)))
                            = (d.info(DeviceInfo::Image2dMaxWidth), d.info(DeviceInfo::Image2dMaxHeight)) {
                        println!("    max image2D dim: {}x{}", w, h);
                    }
                    if let (Ok(DIR::Image3dMaxWidth(w)), Ok(DIR::Image3dMaxHeight(h)), Ok(DIR::Image3dMaxDepth(d)))
                            = (d.info(DeviceInfo::Image3dMaxWidth), d.info(DeviceInfo::Image3dMaxHeight), d.info(DeviceInfo::Image3dMaxDepth)) {
                        println!("    max image3D dim: {}x{}x{}", w, h, d);
                    }


                    // global memory
                    if let Ok(DIR::GlobalMemSize(size)) = d.info(DeviceInfo::GlobalMemSize) {
                        println!("    global memory size: {}", format_mem(size));
                    }
                    if let Ok(DIR::GlobalMemCacheType(tpe)) = d.info(DeviceInfo::GlobalMemCacheType) {
                        print!("    global memory cache: ");
                        match tpe {
                            DeviceMemCacheType::None => println!("none"),
                            DeviceMemCacheType::ReadOnlyCache => println!("read only"),
                            DeviceMemCacheType::ReadWriteCache => println!("read write")
                        }
                    }
                    if let Ok(DIR::GlobalMemCacheSize(size)) = d.info(DeviceInfo::GlobalMemCacheSize) {
                        println!("    global memory cache size: {}", format_mem(size));
                    }


                    // local memory
                    if let Ok(DIR::LocalMemSize(size)) = d.info(DeviceInfo::LocalMemSize) {
                        println!("    global memory size: {}", format_mem(size));
                    }
                    if let Ok(DIR::LocalMemType(tpe)) = d.info(DeviceInfo::LocalMemType) {
                        print!("    global memory cache: ");
                        match tpe {
                            DeviceLocalMemType::None => println!("none"),
                            DeviceLocalMemType::Local => println!("local"),
                            DeviceLocalMemType::Global => println!("global")
                        }
                    }


                    // constant buffers
                    if let Ok(DIR::MaxConstantBufferSize(size)) = d.info(DeviceInfo::MaxConstantBufferSize) {
                        println!("    max constant buffer size: {}", format_mem(size));
                    }
                    if let Ok(DIR::MaxConstantArgs(n)) = d.info(DeviceInfo::MaxConstantArgs) {
                        println!("    max constant buffers argument: {}", n);
                    }
                }
            }
        } else {
            println!("    {}No devices found on this platform.{}", RED, CLEAR);
        }

        println!();
    }
}