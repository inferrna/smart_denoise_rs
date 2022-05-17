extern crate core;

use std::fs::File;
use std::io::{BufWriter, Cursor};
use std::path::Path;
use std::sync::Arc;
use byteorder::{BigEndian, ReadBytesExt};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::device::{DeviceCreateInfo, Features, QueueCreateInfo};
use vulkano::device::DeviceExtensions;
use vulkano::instance::{InstanceCreateInfo, InstanceExtensions};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBuffer};
use vulkano::device::{Device, Queue};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::format::Format;
use vulkano::image::{ImageCreateFlags, ImageDimensions, ImageUsage, StorageImage};
use vulkano::instance::Instance;
use vulkano::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode, SamplerReductionMode};
use vulkano::sync::GpuFuture;
use vulkano::Version;
use png::{BitDepth, ColorType};
use smart_denoise::{denoise, DenoiseParams, UsingShader};

fn main() {

    let self_name = std::env::args().nth(0).unwrap();
    let usage_str = format!("Usage:\n{} filename_in filename_out fragment|compute [sigma, kSigma, threshold]", self_name);
    let filename_in = std::env::args().nth(1).expect(&format!("{}\n no filename_in found", usage_str));
    let filename_out = std::env::args().nth(2).expect(&format!("{}\n no filename_out found", usage_str));
    let shader_type_str = std::env::args().nth(3).expect(&format!("{}\n no shader type provided. Use fragment or compute", usage_str));

    let denoise_params: DenoiseParams = match std::env::args().nth(4) {
        None => DenoiseParams::default(),
        Some(sigma_str) => {
            let sigma: f32 = sigma_str.parse().expect("Provide floating point value for sigma pls");
            let kSigma: f32 = std::env::args().nth(5)
                .expect("Sigma, provided, but no kSigma found.")
                .parse()
                .expect("Provide floating point value for kSigma pls");
            let threshold: f32 = std::env::args().nth(6)
                .expect("Sigma and kSigma, provided, but no threshold found.")
                .parse()
                .expect("Provide floating point value for threshold pls");
            DenoiseParams::new(sigma, kSigma, threshold)
        }
    };

    let shader_type = match shader_type_str.as_str() {
        "fragment" => UsingShader::Fragment,
        "compute" => UsingShader::Compute,
        v => panic!("Unknown shader type {}. Use fragment or compute.", v)
    };

    let img_file = File::open(filename_in).unwrap();
    let mut decoder = png::Decoder::new(img_file);
    decoder.set_transformations(png::Transformations::IDENTITY);
    let mut reader = decoder.read_info().unwrap();
    let info = reader.info();
    let color_samples = info.color_type.samples() as u32;
    let (img_w, img_h) = info.size();

    let path_out = Path::new(&filename_out);
    let file = File::create(path_out).unwrap();
    let w = BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, img_w, img_h); // Width is 2 pixels and height is 1.
    encoder.set_color(info.color_type);
    encoder.set_depth(info.bit_depth);

    let bytes_pp = match info.bit_depth {
        BitDepth::Eight => 1,
        BitDepth::Sixteen => 2,
        _ => unimplemented!()
    };

    let mut buffer = vec![0; (img_w * img_h * color_samples * bytes_pp) as usize];

    let mut writer = encoder.write_header().unwrap();

    match info.bit_depth {
        BitDepth::Eight => {
            reader.next_frame(&mut buffer).unwrap();
            let result_bytes = denoise(&buffer, img_w, img_h, shader_type, denoise_params);
            writer.write_image_data(&result_bytes.into_iter().collect::<Vec<u8>>());
        }
        BitDepth::Sixteen => {
            reader.next_frame(&mut buffer).unwrap();
            let mut buffer_u16 = vec![0; (img_h * img_w * color_samples) as usize];
            let mut buffer_cursor = Cursor::new(buffer);
            buffer_cursor
                .read_u16_into::<BigEndian>(&mut buffer_u16)
                .unwrap();
            let result_bytes = denoise(&buffer_u16, img_w, img_h, shader_type, denoise_params);
            let mut buffer_u8_le = result_bytes.into_iter().map(|v: u16| v.to_be_bytes()).flatten().collect::<Vec<u8>>();
            writer.write_image_data(&buffer_u8_le);

        }
        v => {panic!("Supposed to be 8 or 16 bit image, got {:?}", v)}
    }
}
