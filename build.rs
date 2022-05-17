mod templates;

fn main() {
    // Tell Cargo that if the given file changes, to rerun this build script.
    //println!("cargo:rerun-if-changed=src/templates/mul_shaders.rs");
    // Use the `cc` crate to build a C file and statically link it.
    templates::generate_shaders();
}