extern crate ocl;
use ocl::prm::Char;
use ocl::ProQue;
use godot::prelude::*;
use godot::engine::Node;
use godot::engine::INode;
use godot::engine::Image;
use godot::engine::image::Format;

struct MyExtension;

#[gdextension]
unsafe impl ExtensionLibrary for MyExtension {}

#[derive(GodotClass)]
#[class(base=Node, init)]
struct MandelbrotImageBuilder {
    base: Base<Node>
}

#[godot_api]
impl INode for MandelbrotImageBuilder {
}

#[godot_api]
impl MandelbrotImageBuilder {
    // #[func]
    // fn create_image(&mut self, width: i32, height: i32) -> Gd<Image> {
    //     let mut data = PackedByteArray::new();
    //     match compute(width, height) {
    //         Ok(v) => for i in 0..(width * height * 3).try_into().unwrap() {data.push(v[i]);},
    //         Err(e) => panic!("{e}")
    //     }

    //     if data.len() > ((width * height * 3)).try_into().unwrap() {
    //         godot_print!("How.");
    //     }

    //     let image = Image::create_from_data(width, height, false, Format::RGB8, data);
    //     image.expect("No image was created.")
    // }

    #[func]
    fn create_image_data(&mut self, width: i32, height: i32) -> PackedByteArray {
        let mut data = PackedByteArray::new();
        match compute(width, height) {
            Ok(v) => for i in v {data.push(i);},
            Err(e) => panic!("{e}")
        }

        data
    }
}

fn compute(width: i32, height: i32) -> ocl::Result<Vec<u8>> {
    let src = r#"
        __kernel void mandelbrot(__global char* buffer, char scalar) {
            buffer[get_global_id(0)] = scalar;
        }
    "#;

    let pro_que = ProQue::builder().src(src).dims(width * height * 3).build()?;

    let buffer = pro_que.create_buffer::<u8>()?;

    let kernel = pro_que
        .kernel_builder("mandelbrot")
        .arg(&buffer)
        .arg(&Char::new(0xFFu8 as i8))
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    let mut vec = vec![0u8; buffer.len()];
    buffer.read(&mut vec).enq()?;

    Ok(vec)
}
