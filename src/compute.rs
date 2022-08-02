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


use std::collections::HashMap;
use std::rc::Rc;
use std::cell::{RefCell, RefMut, Ref};

use ocl::{ProQue, Buffer};

use rhai::{Engine, Dynamic, Scope, AST};

use image::RgbImage;


pub struct CInstance {
    rhai_eng: Engine,
    rhai_ast: AST,
    scope: CScope
}


impl CInstance {


    pub fn init(verbose: bool, ocl_prog: String, pipeline: String, size: (usize, usize)) -> Self {
        if verbose {
            println!("* Initializing compute environment");
            println!("** Reading opencl source");
        }

        let mut ocl_src = String::new();
        {
            use std::io::{BufReader, Read};
            use std::fs::File;

            let mut f = BufReader::new(File::open(&ocl_prog).expect(
                format!("Could not read file {}", ocl_prog).as_str()
            ));
            f.read_to_string(&mut ocl_src).unwrap();
        }

        if verbose {
            println!("** Creating queue");
        }

        let prog_queue = ProQue::builder()
            .src(ocl_src)
            .dims(size)
            .build()
            .expect("Could not create the OpenCL queue.");


        if verbose {
            println!("** Creating io buffers");
        }

        let mut buffers = HashMap::new();


        buffers.insert("input".into(), Buff::DynImage(Buffer::<u8>::builder()
            .queue(prog_queue.queue().clone())
            .len(size.0 * size.1 * 3)
            .build()
            .expect("Could not allocate buffer")));


        buffers.insert("output".into(), Buff::DynImage(Buffer::<u8>::builder()
            .queue(prog_queue.queue().clone())
            .len(size.0 * size.1 * 3)
            .build()
            .expect("Could not allocate buffer")));
        

        if verbose {
            println!("* Initializing pipeline");
            println!("** Creating Rhai environment");
        }


        let mut cscope = CScope::init(buffers, prog_queue);
        cscope.set_image_size(size);

        let mut rhai_eng = Engine::new();

        rhai_eng.register_type_with_name::<CScope>("Ocl")
            .register_fn("call_kernel", CScope::call_kernel);

        rhai_eng.register_type_with_name::<BufferRhaiRef>("Buffer")
            .register_fn("len", BufferRhaiRef::len);
        rhai_eng.register_type_with_name::<ImageRhaiRef>("Image")
            .register_fn("width", ImageRhaiRef::width)
            .register_fn("height", ImageRhaiRef::height);

        
        if verbose {
            println!("** Compiling rhai code");
        }

        let rhai_ast = rhai_eng.compile_file(pipeline.into()).unwrap();


        if verbose {
            println!("** Running initializing code");
        }

        { // script initialization
            let mut init_eng = Engine::new();
            let mut init_scope = Scope::new();

            init_eng.register_type_with_name::<CScope>("Ocl")
                .register_fn("create_int_buffer", CScope::create_int_buffer)
                .register_fn("create_float_buffer", CScope::create_float_buffer)
                .register_fn("create_dynimage", CScope::create_dynimage);

            init_scope.push("ocl", cscope.clone());

            let _result: () = init_eng.call_fn(&mut init_scope, &rhai_ast, "init", ()).unwrap();
        }


        if verbose {
            println!("Finished initialization.");
        }
        Self {
            rhai_eng: rhai_eng,
            rhai_ast: rhai_ast,
            scope: cscope
        }
    }


    pub fn compute(&mut self, img: &RgbImage) -> RgbImage {
        self.scope.set_image_size((img.width() as usize, img.height() as usize));
        self.scope.set_input(img);
        let mut scope = self.scope.create_rhai_scope();
        scope.push("ocl", self.scope.clone());
        scope.push_constant("IMG_WIDTH", img.width())
            .push_constant("IMG_HEIGTH", img.height());

        let _result: () = self.rhai_eng.call_fn(&mut scope, &self.rhai_ast, "run", ()).unwrap();

        return self.scope.get_output();
    }

}


#[derive(Clone)]
struct CScope {
    buffers: Rc<RefCell<HashMap<String, Buff>>>,
    prog_queue: ProQue,
    dynimg_size: (usize, usize)
}


/// Differenciate between general buffers and images.
/// In the code, general buffers will be sent to opencl as is,
/// but images will be sent with their dimentions (they take three arguments)
#[derive(Clone)]
enum Buff {
    IntBuffer(Buffer<i64>),
    FloatBuffer(Buffer<f64>),
    DynImage(Buffer<u8>),
    Image(Buffer<u8>, usize, usize)
}


#[derive(Clone)]
struct BufferRhaiRef {
    name: String,
    size: usize
}


// TODO: allow modifications
impl BufferRhaiRef {

    fn len(&self) -> usize {
        self.size
    }
}


// TODO: allow modifications
#[derive(Clone)]
struct ImageRhaiRef {
    name: String,
    width: usize,
    height: usize
}


impl ImageRhaiRef {

    fn width(&self) -> usize {
        self.width
    }


    fn height(&self) -> usize {
        self.height
    }
}


impl CScope {


    fn init(buffers: HashMap<String, Buff>, prog_queue: ProQue) -> Self {
        Self {
            buffers: Rc::new(RefCell::new(buffers)),
            prog_queue: prog_queue,
            dynimg_size: (0, 0)
        }
    }


    fn call_kernel(&mut self, name: String, args: Vec<Dynamic>) {
        let mut ker = self.prog_queue.kernel_builder(&name);

        for arg in args {
            macro_rules! add_arg {
                (type $t:ty) => {
                    if arg.is::<$t>() { ker.arg(arg.cast::<$t>()); continue; }
                };
                (vect $t:ty) => { // TODO: use when it works
                    add_arg!(type $t);
                    add_arg!(type [$t; 2]);
                    add_arg!(type [$t; 3]);
                    add_arg!(type [$t; 4]);
                    add_arg!(type [$t; 8]);
                    add_arg!(type [$t; 16]);
                };
            }
            macro_rules! add_args {
                ($($t:ty as $($mod:ident)?),+) => {
                    $( add_arg!($($mod)? $t); )+
                }
            }

            add_args!(i8 as type, u8 as type, i16 as type, u16 as type,
                i32 as type, u32 as type, i64 as type, u64 as type, f32 as type,
                f64 as type, isize as type, usize as type);
            
            if arg.is::<BufferRhaiRef>() {
                let buff = arg.cast::<BufferRhaiRef>();

                if !self.get_buffers().contains_key(&buff.name) {
                    panic!("There is no buffer named {}", buff.name);
                }
                
                match &self.get_buffers()[&buff.name] {
                    Buff::IntBuffer(b) => {
                        ker.arg(b.clone());
                    }
                    Buff::FloatBuffer(b) => {
                        ker.arg(b.clone());
                    }
                    _ => { panic!("There is no buffer named {}", buff.name); }
                }

                continue;
            }

            if arg.is::<ImageRhaiRef>() {
                let img = arg.cast::<ImageRhaiRef>();

                if !self.get_buffers().contains_key(&img.name) {
                    panic!("There is no image named {}", img.name);
                }

                match &self.get_buffers()[&img.name] {
                    Buff::Image(b, _, _) => {
                        ker.arg(b.clone()).arg(img.width).arg(img.height);
                    },
                    Buff::DynImage(b) => {
                        ker.arg(b.clone());
                    }
                    _ => { panic!("There is no image named {}", img.name); }
                }

                continue;
            }
        }

        let ker = ker.arg(self.dynimg_size.0 as u32)
            .arg(self.dynimg_size.1 as u32)
            .build()
            .expect("Could not build kernel.");


        unsafe {
            ker.enq().expect("Could not run kernel.");
        }
    }


    fn get_buffers(&self) -> Ref<'_, HashMap<String, Buff>> {
        self.buffers.borrow()
    }

    fn get_buffers_mut(&mut self) -> RefMut<'_, HashMap<String, Buff>> {
        self.buffers.borrow_mut()
    }


    fn set_image_size(&mut self, size: (usize, usize)) {
        self.dynimg_size = size;
    }


    // TODO: more error checks with set and get image
    fn set_input(&mut self, img: &RgbImage) {
        if let Buff::DynImage(buff) = &self.get_buffers()["input".into()] {
            buff.write(img.as_raw()).enq().unwrap();
        }
    }


    fn get_output(&self) -> RgbImage {
        let mut pixels = vec![0u8; self.dynimg_size.0 * self.dynimg_size.1 * 3];
        if let Buff::DynImage(buff) = &self.get_buffers()["output".into()] {
            buff.read(&mut pixels).enq().unwrap(); // TODO: pixels having the wrong dimentions due to direct call to read
        }
        let rgb_image = RgbImage::from_raw(self.dynimg_size.0 as u32, self.dynimg_size.1 as u32, pixels).unwrap();
        return rgb_image;
    }


    fn create_rhai_scope(&self) -> Scope {
        let mut scope = Scope::new();

        for name in self.get_buffers().keys() {
            match &self.get_buffers()[name] {
                Buff::IntBuffer(b) => {
                    scope.push(name, BufferRhaiRef{name: name.clone(), size: b.len()});
                }
                Buff::FloatBuffer(b) => {
                    scope.push(name, BufferRhaiRef{name: name.clone(), size: b.len()});
                }
                Buff::DynImage(_) => {
                    scope.push(name, ImageRhaiRef{name: name.clone(), width: self.dynimg_size.0, height: self.dynimg_size.1});
                }
                Buff::Image(_, w, h) => {
                    scope.push(name, ImageRhaiRef{name: name.clone(), width: *w, height: *h});
                }
            }
        }

        return scope;
    }


    fn create_int_buffer(&mut self, name: String, raw_data: Vec<Dynamic>) -> BufferRhaiRef {
        let mut data = Vec::with_capacity(raw_data.len());
        for d in raw_data {
            data.push(d.cast::<i64>());
        }
        
        let buff = Buffer::<i64>::builder()
            .queue(self.prog_queue.queue().clone())
            .len(data.len())
            .build()
            .expect("Could not allocate buffer");
        buff.write(&data).enq().unwrap();
        self.get_buffers_mut().insert(name.clone(), Buff::IntBuffer(buff));
        return BufferRhaiRef {
            name: name,
            size: data.len()
        };
    }


    fn create_float_buffer(&mut self, name: String, raw_data: Vec<Dynamic>) -> BufferRhaiRef {
        let mut data = Vec::with_capacity(raw_data.len());
        for d in raw_data {
            data.push(d.cast::<f64>());
        }
        
        let buff = Buffer::<f64>::builder()
            .queue(self.prog_queue.queue().clone())
            .len(data.len())
            .build()
            .expect("Could not allocate buffer");
        buff.write(&data).enq().unwrap();
        self.get_buffers_mut().insert(name.clone(), Buff::FloatBuffer(buff));

        return BufferRhaiRef {
            name: name,
            size: data.len()
        };
    }


    fn create_dynimage(&mut self, name: String) {
        let queue = self.prog_queue.queue().clone();
        let size = self.dynimg_size.0 * self.dynimg_size.1 * 3;
        self.get_buffers_mut().insert(name, Buff::DynImage(Buffer::<u8>::builder()
            .queue(queue)
            .len(size)
            .build()
            .expect("Could not allocate buffer")));
    }


    fn create_image(&mut self, name: String, width: usize, height: usize) -> ImageRhaiRef {
        let queue = self.prog_queue.queue().clone();
        self.get_buffers_mut().insert(name.clone(), Buff::Image(Buffer::<u8>::builder()
            .queue(queue)
            .len(width * height * 3)
            .build()
            .expect("Could not allocate buffer"), width, height));
        return ImageRhaiRef {
            name: name,
            width: width,
            height: height
        };
    }
}
