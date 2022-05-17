extern crate core;

mod vertex_shader;
mod denoise_compute;
mod denoise_frag;
mod generated;

use std::sync::Arc;
use bytemuck::Pod;
use num_traits::Zero;
use vulkano::buffer::{BufferContents, BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBuffer};
use vulkano::device::{Device, Features, DeviceCreateInfo, Queue, QueueCreateInfo};
use vulkano::device::DeviceExtensions;
use vulkano::instance::{Instance, InstanceCreateInfo, InstanceExtensions};
use vulkano::Version;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::image::{ImageCreateFlags, ImageDimensions, ImageUsage, StorageImage};
use vulkano::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode, SamplerReductionMode};
use vulkano::sync::GpuFuture;

pub fn vlk_init() -> (Arc<Device>, Arc<Queue>) {
        let instance =
            Instance::new(InstanceCreateInfo { application_name: None, application_version: Version::V1_3, enabled_extensions: InstanceExtensions::none(),
                enabled_layers: vec![], engine_name: None, engine_version: Default::default(), function_pointers: None, max_api_version: None, _ne: Default::default() })
                .expect("failed to create instance");

    let device_extensions = DeviceExtensions {
        khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::none()
    };

    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
        .filter_map(|p| {
            p.queue_families()
             .find(|&q| q.supports_compute())
             .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
        })
        .unwrap();

        let devinfo = DeviceCreateInfo {
            enabled_extensions:  DeviceExtensions {
                khr_storage_buffer_storage_class: true,
                ..DeviceExtensions::none()
            },
            enabled_features: Features{
                shader_storage_image_extended_formats: true,
                sampler_anisotropy: false,
                ..Features::none()
            },
            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            _ne: Default::default()
        };

        let (device, mut queues) = {
                Device::new(
                    physical_device,
                        devinfo,
                )
                    .expect("failed to create device")
        };

        //let limits = device.physical_device().limits();


        let queue = queues.next().unwrap();
        (device, queue)
}

#[derive(Debug, Copy, Clone)]
pub enum UsingShader {
    Fragment,
    Compute
}

#[derive(Debug, Copy, Clone)]
pub struct DenoiseParams {
    sigma: f32,
    kSigma: f32,
    threshold: f32
}
#[derive(Debug, Copy, Clone)]
pub struct ShaderParams {
    Width: u32,
    Height: u32,
    sigma: f32,
    kSigma: f32,
    threshold: f32
}

impl ShaderParams {
    pub fn new(Width: u32, Height: u32, denoise_parameters: DenoiseParams) -> Self {
        Self { Width, Height,
            sigma: denoise_parameters.sigma,
            kSigma: denoise_parameters.kSigma,
            threshold: denoise_parameters.threshold }
    }
}

impl DenoiseParams {
    pub fn new(sigma: f32, kSigma: f32, threshold: f32) -> Self {
        Self { sigma, kSigma, threshold }
    }
}

impl Default for DenoiseParams {
    fn default() -> Self {
        Self {
            sigma: 7.0,
            kSigma: 3.0,
            threshold: 0.195
        }
    }
}

pub trait TypeToFormat {
    fn type2result_format(num_samples: usize) -> vulkano::format::Format;
    fn type2sampled_format(num_samples: usize) -> vulkano::format::Format;
}

impl TypeToFormat for f32 {
    fn type2result_format(num_samples: usize) -> vulkano::format::Format {
        match num_samples {
            1 => vulkano::format::Format::R32_SFLOAT,
            3 => vulkano::format::Format::R32G32B32A32_SFLOAT,
            4 => vulkano::format::Format::R32G32B32A32_SFLOAT,
            _ => unimplemented!("Implemented only for single channel floating point."),
        }
    }

    fn type2sampled_format(num_samples: usize) -> vulkano::format::Format {
        match num_samples {
            1 => vulkano::format::Format::R32_SFLOAT,
            3 => vulkano::format::Format::R32G32B32A32_SFLOAT,
            4 => vulkano::format::Format::R32G32B32A32_SFLOAT,
            _ => unimplemented!("Implemented only for single channel floating point."),
        }
    }
}
impl TypeToFormat for u8 {
    fn type2result_format(num_samples: usize) -> vulkano::format::Format {
        match num_samples {
            1 => vulkano::format::Format::R8_UINT,
            3 => vulkano::format::Format::R8G8B8A8_UINT,
            4 => vulkano::format::Format::R8G8B8A8_UINT,
            _ => unimplemented!("Implemented only for 1 & 4 channel u8."),
        }
    }

    fn type2sampled_format(num_samples: usize) -> vulkano::format::Format {
        match num_samples {
            1 => vulkano::format::Format::R32_SFLOAT,
            3 => vulkano::format::Format::R32G32B32A32_SFLOAT,
            4 => vulkano::format::Format::R32G32B32A32_SFLOAT,
            _ => unimplemented!("Implemented only for 1 & 4 channel u8."),
        }
    }
}
impl TypeToFormat for u16 {
    fn type2result_format(num_samples: usize) -> vulkano::format::Format {
        match num_samples {
            1 => vulkano::format::Format::R16_UINT,
            3 => vulkano::format::Format::R16G16B16A16_UINT,
            4 => vulkano::format::Format::R16G16B16A16_UINT,
            _ => unimplemented!("Implemented only for 1 & 4 channel u16."),
        }
    }

    fn type2sampled_format(num_samples: usize) -> vulkano::format::Format {
        match num_samples {
            1 => vulkano::format::Format::R32_SFLOAT,
            3 => vulkano::format::Format::R32G32B32A32_SFLOAT,
            4 => vulkano::format::Format::R32G32B32A32_SFLOAT,
            _ => unimplemented!("Implemented only for 1 & 4 channel u8."),
        }
    }
}

pub fn denoise<D>(buf: &[D], img_w: u32, img_h: u32, shader_type: UsingShader, params: DenoiseParams, use_hsv: bool) -> Vec<D>
where D: TypeToFormat + num_traits::AsPrimitive<f32> + Sized + Copy + Zero + Send + Sync,
      D: Pod,
{
    let (device, queue) = vlk_init();

    let num_input_samples = buf.len() / (img_w * img_h) as usize;

    let num_output_samples = match num_input_samples {
        3 => 4,
        _ => num_input_samples
    };

    let input_usage = BufferUsage{
        transfer_source: true,
        transfer_destination: false,
        uniform_texel_buffer: false,
        storage_texel_buffer: false,
        uniform_buffer: false,
        storage_buffer: true,
        index_buffer: false,
        vertex_buffer: false,
        indirect_buffer: false,
        device_address: true,
        _ne: Default::default()
    };

    let input2sample: Vec<f32> = match num_input_samples {
        3 => buf.iter()
                .enumerate()
                .flat_map(|(i, x)| {
                    let rs: f32 = (*x).as_();
                    if i%3==2 {vec![rs, 0.0]} //Adds alpha channel for each 3rd item
                    else {vec![rs]}
                })
                .collect(),
        _ => buf.iter()
                .map(|x| {
                    let rs: f32 = (*x).as_();
                    rs
                })
                .collect()
    };

    let img_buf: Arc<CpuAccessibleBuffer<[f32]>> =
            CpuAccessibleBuffer::from_iter(device.clone(), input_usage, false,
                                           input2sample.into_iter()).expect("failed to create buffer");

    dbg!(D::type2sampled_format(num_input_samples));
    let input_img = StorageImage::with_usage(device.clone(),
                                             ImageDimensions::Dim2d { width: img_w, height: img_h, array_layers: 1},
                                             D::type2sampled_format(num_input_samples),
                                             ImageUsage {
                                                 transfer_source: false,
                                                 transfer_destination: true,
                                                 sampled: true,
                                                 storage: true,
                                                 color_attachment: false,
                                                 depth_stencil_attachment: false,
                                                 transient_attachment: false,
                                                 input_attachment: false
                                             },
                                             ImageCreateFlags::none(),
                                             Some(queue.family())).unwrap();


    //Buffer to image
    let command_buffer = {
        let mut builder =
            AutoCommandBufferBuilder::primary(device.clone(), queue.family(), CommandBufferUsage::OneTimeSubmit).unwrap();
        builder
            .copy_buffer_to_image(img_buf.clone(), input_img.clone()).unwrap();
        builder.build().unwrap()
    };
    println!("Execute copy buffer to image");
    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished.then_signal_fence_and_flush().unwrap()
            .wait(None).unwrap();

    let sampler = Sampler::new(device.clone(), SamplerCreateInfo {
        mag_filter: Filter::Linear,
        min_filter: Filter::Linear,
        mipmap_mode: SamplerMipmapMode::Linear,
        reduction_mode: SamplerReductionMode::WeightedAverage,
        address_mode: [SamplerAddressMode::Repeat; 3],
        mip_lod_bias: 0.0,
        lod: 0.0..=0.0,
        anisotropy: None,
        //anisotropy: Some(4.0),
        ..Default::default()
    }).unwrap();

    let result_img = StorageImage::with_usage(device.clone(),
                                              ImageDimensions::Dim2d { width: img_w, height: img_h, array_layers: 1},
                                              D::type2result_format(num_output_samples),
                                              ImageUsage {
                                                  transfer_source: true,
                                                  transfer_destination: false,
                                                  sampled: false,
                                                  storage: true,
                                                  color_attachment: true,
                                                  depth_stencil_attachment: false,
                                                  transient_attachment: false,
                                                  input_attachment: false
                                              },
                                              ImageCreateFlags::none(),
                                              Some(queue.family())).unwrap();

    match shader_type {
        UsingShader::Fragment => denoise_frag::denoise(device.clone(), queue.clone(), input_img.clone(),
                                                       result_img.clone(), sampler.clone(), params, use_hsv),
        UsingShader::Compute => denoise_compute::denoise(device.clone(), queue.clone(), input_img.clone(),
                                                         result_img.clone(), sampler.clone(), params, use_hsv)
    }

    let result_buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage{
        transfer_source: false,
        transfer_destination: true,
        uniform_texel_buffer: false,
        storage_texel_buffer: false,
        uniform_buffer: true,
        storage_buffer: true,
        index_buffer: false,
        vertex_buffer: false,
        indirect_buffer: false,
        device_address: true,
        _ne: Default::default()
    }, false, (0..img_w*img_h*num_output_samples as u32).map(|_| D::zero())).expect("failed to create output buffer");

    //Read filtered image
    let command_buffer = {
        let mut builder =
            AutoCommandBufferBuilder::primary(device.clone(), queue.family(), CommandBufferUsage::OneTimeSubmit).unwrap();
        builder
            .copy_image_to_buffer(result_img.clone(), result_buf.clone()).unwrap();
        builder.build().unwrap()
    };
    println!("Execute copy image to buffer");
    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished.then_signal_fence_and_flush().unwrap()
            .wait(None).unwrap();

    let rl = result_buf.read().unwrap();
    match num_input_samples {
        3 => rl.chunks(4).into_iter().flat_map(|v| vec![v[0],v[1],v[2]]).collect(),
        _ => rl.to_vec()
    }
}
