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
use clap::Parser;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Path to the input file. Only png is currently accepted
    #[clap(short, long)]
    filename_in: String,

    /// Path to the input png file. Will use same format as the input one
    #[clap(long)]
    filename_out: String,

    /// Which shader type to use
    #[clap(long)]
    shader_type: String,

    ///Sigma parameter
    #[clap(long)]
    sigma: Option<f32>,

    ///kSigma parameter
    #[clap(long)]
    kSigma: Option<f32>,

    ///threshold parameter
    #[clap(long)]
    threshold: Option<f32>,

    ///Process denoise in HSV (H & V actually) space
    #[clap(long)]
    use_hsv: bool,
}

fn main() {
    let args = Args::parse();

    let denoise_params: DenoiseParams = if args.sigma.is_none() && args.kSigma.is_none() && args.threshold.is_none() {
        DenoiseParams::default()
    } else {
        DenoiseParams::new(args.sigma.expect("Provide all 3 parameters: sigma, kSigma and threshold"),
                           args.kSigma.expect("Provide all 3 parameters: sigma, kSigma and threshold"),
                           args.threshold.expect("Provide all 3 parameters: sigma, kSigma and threshold"))
    };

    let shader_type = match args.shader_type.as_str() {
        "fragment" => UsingShader::Fragment,
        "compute" => UsingShader::Compute,
        v => panic!("Unknown shader type {}. Use fragment or compute.", v)
    };

    let img_file = File::open(args.filename_in).unwrap();
    let mut decoder = png::Decoder::new(img_file);
    decoder.set_transformations(png::Transformations::IDENTITY);
    let mut reader = decoder.read_info().unwrap();
    let info = reader.info();
    let color_samples = info.color_type.samples() as u32;
    let (img_w, img_h) = info.size();

    let path_out = Path::new(&args.filename_out);
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
            let result_bytes = denoise(&buffer, img_w, img_h, shader_type, denoise_params, args.use_hsv);
            writer.write_image_data(&result_bytes.into_iter().collect::<Vec<u8>>());
        }
        BitDepth::Sixteen => {
            reader.next_frame(&mut buffer).unwrap();
            let mut buffer_u16 = vec![0; (img_h * img_w * color_samples) as usize];
            let mut buffer_cursor = Cursor::new(buffer);
            buffer_cursor
                .read_u16_into::<BigEndian>(&mut buffer_u16)
                .unwrap();
            let result_bytes = denoise(&buffer_u16, img_w, img_h, shader_type, denoise_params, args.use_hsv);
            let mut buffer_u8_le = result_bytes.into_iter().map(|v: u16| v.to_be_bytes()).flatten().collect::<Vec<u8>>();
            writer.write_image_data(&buffer_u8_le);

        }
        v => {panic!("Supposed to be 8 or 16 bit image, got {:?}", v)}
    }
}
