//! SolidCraft: offscreen recording of the root window.
//!
//! While a [`Target`] is installed ([`start`]), [`crate::winit::Painter`] draws each frame of the
//! root viewport into a texture of its own (at the target's pixels per point) instead of the
//! window's surface, reads it back and hands the pixels to the target's sink. The surface is not
//! touched, so a covered window (whose drawable macOS withholds) still yields every frame. The
//! sink may block (a full queue): the frame then waits, nothing is dropped.

use std::sync::Mutex;

/// One recorded frame: straight RGBA, `width` × `height`, rows top to bottom.
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Where recorded frames go.
pub struct Target {
    /// Pixels per point of the recorded frames; `None` for the window's own.
    pub pixels_per_point: Option<f32>,
    /// Called once per painted frame, on the window thread.
    pub sink: Box<dyn FnMut(Frame) + Send>,
}

static TARGET: Mutex<Option<Target>> = Mutex::new(None);

fn target() -> std::sync::MutexGuard<'static, Option<Target>> {
    TARGET.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Record every frame painted from now on into `target` (replacing any earlier one).
pub fn start(target_: Target) -> Option<Target> {
    target().replace(target_)
}

/// Stop recording; the target that was installed.
pub fn stop() -> Option<Target> {
    target().take()
}

/// Whether frames are being recorded.
pub fn active() -> bool {
    target().is_some()
}

/// The pixels per point the installed target asks for.
pub(crate) fn pixels_per_point() -> Option<Option<f32>> {
    target().as_ref().map(|t| t.pixels_per_point)
}

/// Hand a frame to the installed target, if any.
pub(crate) fn deliver(frame: Frame) {
    if let Some(t) = target().as_mut() {
        (t.sink)(frame);
    }
}

/// The offscreen textures one recording size needs, kept between frames.
pub(crate) struct Textures {
    pub size: (u32, u32),
    pub color: wgpu::Texture,
    pub msaa: Option<wgpu::TextureView>,
    pub depth: Option<wgpu::TextureView>,
    pub buffer: wgpu::Buffer,
    pub padded_row: u32,
}

impl Textures {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        samples: u32,
        depth_format: Option<wgpu::TextureFormat>,
        size: (u32, u32),
    ) -> Self {
        let extent = wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        };
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("egui_record_texture"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let msaa = (samples > 1).then(|| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("egui_record_msaa"),
                    size: extent,
                    mip_level_count: 1,
                    sample_count: samples,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[format],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        });
        let depth = depth_format.map(|depth_format| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("egui_record_depth"),
                    size: extent,
                    mip_level_count: 1,
                    sample_count: samples.max(1),
                    dimension: wgpu::TextureDimension::D2,
                    format: depth_format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[depth_format],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        });
        let padded_row = wgpu::util::align_to(size.0 * 4, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("egui_record_buffer"),
            size: u64::from(padded_row) * u64::from(size.1),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            size,
            color,
            msaa,
            depth,
            buffer,
            padded_row,
        }
    }

    /// Wait for the copied frame and read it back as straight RGBA.
    pub fn read(&self, device: &wgpu::Device) -> Option<Frame> {
        let swizzle = match self.color.format() {
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => [0, 1, 2, 3],
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => [2, 1, 0, 3],
            other => {
                log::error!("record: cannot read back frames of format {other:?}");
                return None;
            }
        };
        let slice = self.buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        if let Err(e) = device.poll(wgpu::PollType::wait_indefinitely()) {
            log::error!("record: waiting for the frame failed: {e}");
        }
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                log::error!("record: mapping the frame failed: {e}");
                return None;
            }
            Err(_) => return None,
        }
        let (w, h) = self.size;
        let mut rgba = Vec::with_capacity((w as usize) * (h as usize) * 4);
        {
            let mapped = match slice.get_mapped_range() {
                Ok(m) => m,
                Err(e) => {
                    log::error!("record: reading the frame failed: {e}");
                    self.buffer.unmap();
                    return None;
                }
            };
            let row_bytes = (w as usize) * 4;
            for row in mapped.chunks(self.padded_row as usize).take(h as usize) {
                for px in row.get(..row_bytes).unwrap_or(&[]).chunks_exact(4) {
                    // Opaque output: the window has no transparency worth keeping.
                    rgba.extend_from_slice(&[px[swizzle[0]], px[swizzle[1]], px[swizzle[2]], 255]);
                }
            }
        }
        self.buffer.unmap();
        Some(Frame {
            width: w,
            height: h,
            rgba,
        })
    }
}
