use handlebars::Handlebars;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

pub fn generate_shaders() {
    let mut handlebars = Handlebars::new();
    handlebars
        .register_template_file("shaders", "./templates/denoise_shader.mustache")
        .unwrap();


    let mut datas = vec![];

    {
        let mut data = HashMap::new();
        //8-bit grayscale
        data.insert("output_format", "r8ui");
        data.insert("max_val", "255");
        data.insert("swizzle_vec", "r");
        data.insert("processing_t", "float");
        data.insert("output_t", "uvec4");
        data.insert("is_int_type", "true");
        datas.push(data);
    }

    {
        let mut data = HashMap::new();
        //16-bit grayscale
        data.insert("output_format", "r16ui");
        data.insert("max_val", "255");
        data.insert("swizzle_vec", "r");
        data.insert("processing_t", "float");
        data.insert("output_t", "uvec4");
        data.insert("is_int_type", "true");
        datas.push(data);
    }

    {
        let mut data = HashMap::new();
        //8-bit rgba
        data.insert("output_format", "rgba8ui");
        data.insert("max_val", "255");
        data.insert("swizzle_vec", "rgba");
        data.insert("processing_t", "vec4");
        data.insert("output_t", "uvec4");
        data.insert("is_int_type", "true");
        data.insert("is_vector_type", "true");
        datas.push(data);
    }


    {
        let mut data = HashMap::new();
        //16-bit rgba
        data.insert("output_format", "rgba16ui");
        data.insert("max_val", "65535");
        data.insert("swizzle_vec", "rgba");
        data.insert("processing_t", "vec4");
        data.insert("output_t", "uvec4");
        data.insert("is_int_type", "true");
        data.insert("is_vector_type", "true");
        datas.push(data);
    }

    {
        let mut data = HashMap::new();
        //32-bit float
        data.insert("output_format", "r32f");
        //data.insert("max_val", "1.0"); //Assuming data is already normalized
        data.insert("swizzle_vec", "r");
        data.insert("processing_t", "float");
        data.insert("output_t", "vec4");
        datas.push(data);
    }

    let format_pairs: HashMap::<String, String>  = vec![
        ("r8ui".to_string(), "R8_UINT".to_string()),
        ("r16ui".to_string(), "R16_UINT".to_string()),
        ("rgba8ui".to_string(), "R8G8B8A8_UINT".to_string()),
        ("r32f".to_string(), "R32_SFLOAT".to_string()),
        ("rgba16ui".to_string(), "R16G16B16A16_UINT".to_string())]
        .into_iter()
        .collect();

    let type_pairs = vec![("compute", "UsingShader::Compute"), ("fragment", "UsingShader::Fragment")];

    let mut shader_mods = vec![];
    let mut shader_matchers = vec!["\n".to_string()];

    let base_path = "src/generated";
    for (shadert, shadert_enum_val) in type_pairs.into_iter() {

        for d in datas.iter_mut() {
            d.insert(shadert, "true");
            let shader = handlebars.render("shaders", d).unwrap();

            let format = d.get("output_format").unwrap();
            let vk_type = format_pairs.get(*format).unwrap();

            let name = format!("denoise_shader_{}_{}", shadert, format);
            shader_mods.push(format!("pub(crate) mod {};", &name));
            let align = 40-(vk_type.len()+shadert_enum_val.len());
            shader_matchers.push(format!("{:>12}(Format::{}, {}) => {:align$}{}::load(device.clone()),", "", vk_type, shadert_enum_val, "", &name));
            let mut file = File::create(format!("{}/{}.rs", base_path, &name)).unwrap();
            file.write_all(shader.as_bytes()).unwrap();
            d.remove(shadert);
        }
    }

    let mut file = File::create(format!("{}/mod.rs", base_path)).unwrap();
    file.write_all(shader_mods.join("\n").as_bytes()).unwrap();

    file.write_all(r#"

use std::sync::Arc;
use vulkano::format::Format;
use vulkano::device::Device;
use vulkano::shader::ShaderModule;
use crate::UsingShader;

    pub(crate) fn get_denoise_shader(device: Arc<Device>, format: Format, shader_type: UsingShader) -> Arc<ShaderModule> {
        match (format, shader_type) {"#.as_bytes()).unwrap();

    file.write_all(shader_matchers.join("\n").as_bytes()).unwrap();

    file.write_all(r#"
            _ => unimplemented!(),
        }.unwrap()
    }
    "#.as_bytes()).unwrap();

}