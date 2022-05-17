use std::sync::Arc;
use std::time::Instant;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, Queue};
use vulkano::image::{ImageAccess, StorageImage};
use vulkano::image::view::ImageView;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, Subpass};
use vulkano::sampler::Sampler;
use vulkano::sync;
use vulkano::sync::GpuFuture;
use bytemuck::{Pod, Zeroable};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess};
use crate::{DenoiseParams, ShaderParams, UsingShader};


#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
struct Vertex {
    position: [f32; 2],
}

vulkano::impl_vertex!(Vertex, position);

pub(crate) fn denoise(device: Arc<Device>, queue: Arc<Queue>, input_img: Arc<StorageImage>, result_img: Arc<StorageImage>,
                      sampler: Arc<Sampler>, denoise_params: DenoiseParams, use_hsv: bool) {
    let shader = crate::generated::get_denoise_shader(device.clone(), result_img.format(), UsingShader::Fragment, use_hsv);
    let vert_shader = crate::vertex_shader::load(device.clone()).unwrap();


    let img_w = input_img.dimensions().width();
    let img_h = input_img.dimensions().height();

    let render_pass = vulkano::single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: DontCare,
                    format: result_img.format(),
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        )
        .unwrap();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [img_w as f32, img_h as f32],
        depth_range: 0.0..1.0,
    };

    let graphics_pipeline = GraphicsPipeline::start()
        .fragment_shader(shader.entry_point("main").unwrap(), ())
        .vertex_shader(vert_shader.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport.clone()]))
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap();

    let input_view = ImageView::new_default(input_img.clone()).unwrap();
    let output_view = ImageView::new_default(result_img.clone()).unwrap();

    let layout = graphics_pipeline.layout().set_layouts().get(0).unwrap();

    let items = [WriteDescriptorSet::image_view_sampler(0, input_view.clone(), sampler.clone()),];
        //WriteDescriptorSet::image_view(1, output_view.clone())];

    let set = PersistentDescriptorSet::new(layout.clone(), items).unwrap();

    let push_constants = ShaderParams::new(img_w, img_h, denoise_params);

    let vertices = [
        Vertex {
            position: [-1.0, -1.0],
        },
        Vertex {
            position: [-1.0, 1.0],
        },
        Vertex {
            position: [1.0, -1.0],
        },
        Vertex {
            position: [-1.0, 1.0],
        },
        Vertex {
            position: [1.0, -1.0],
        },
        Vertex {
            position: [1.0, 1.0],
        },
    ];
    let vertex_buffer = CpuAccessibleBuffer::<[Vertex]>::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        vertices,
    )
        .unwrap();


    let framebuffer = Framebuffer::new(
        render_pass.clone(),
        FramebufferCreateInfo {
            attachments: vec![output_view],
            ..Default::default()
        },
    ).unwrap();

    //Computation itself
    let now = Instant::now();
    let command_buffer = {
        let mut builder =
            AutoCommandBufferBuilder::primary(device.clone(), queue.family(), CommandBufferUsage::OneTimeSubmit).unwrap();
        builder
            .begin_render_pass(
                framebuffer.clone(),
                SubpassContents::Inline,
                vec![[0u32, 0u32].into()],
            )
            .unwrap()
            //.bind_pipeline_compute(compute_pipeline.clone())
            .set_viewport(0, [viewport.clone()])
            .bind_pipeline_graphics(graphics_pipeline.clone())
            .bind_descriptor_sets(PipelineBindPoint::Graphics, graphics_pipeline.layout().clone(), 0, set.clone())
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .push_constants(graphics_pipeline.layout().clone(), 0, push_constants)
            .draw(vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap()
            .end_render_pass()
            //.dispatch([img_w, img_h, 1])
            .unwrap();
        builder.build().unwrap()
    };


    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    future.wait(None).unwrap();
    println!("Execute computation itself taken {} milliseconds", now.elapsed().as_millis());
}