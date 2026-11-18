//! SolidCraft: frame-stepped capture. With a capture size set, the root window's frames are drawn
//! into a texture of that size instead of the window's surface, independent of the window (its
//! size, whether it is shown at all); only frames the app arms are drawn and read back, each
//! handed to the sink in order. Frames in between are not drawn: their texture changes wait for
//! the next drawn frame, as they do for a frame without a surface texture.
//!
//! The app arms one frame at a time ([`arm`]) before the frame runs; the painter draws it,
//! reads it back (waiting for the GPU: frames come out in order and complete) and counts it
//! ([`captured`]).

use std::sync::Mutex;

/// One frame read back: rows of `width` pixels, 4 bytes each, no padding, in `format`'s order
/// (`Bgra8Unorm` on macOS: blue, green, red, alpha), as the window would have shown them.
pub struct Frame<'a> {
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
    pub bytes: &'a [u8],
}

/// Where captured frames go. Called on the window thread, in frame order.
pub type Sink = Box<dyn FnMut(Frame<'_>) + Send>;

#[derive(Default)]
struct State {
    size: Option<(u32, u32)>,
    armed: bool,
    captured: u64,
    sink: Option<Sink>,
}

static STATE: Mutex<State> = Mutex::new(State { size: None, armed: false, captured: 0, sink: None });

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Draw the root window's frames at `size` pixels (offscreen), or into the window again (`None`).
pub fn set_size(size: Option<(u32, u32)>) {
    state().size = size;
}

/// The capture size, while frames are drawn offscreen.
pub fn size() -> Option<(u32, u32)> {
    state().size
}

/// Draw and capture the next frame of the root window.
pub fn arm() {
    state().armed = true;
}

/// Whether a frame is armed and not drawn yet.
pub fn armed() -> bool {
    state().armed
}

/// Frames captured so far.
pub fn captured() -> u64 {
    state().captured
}

/// Where the next captured frames go (`None`: nowhere; they are still counted).
pub fn set_sink(sink: Option<Sink>) {
    state().sink = sink;
}

/// Taken by the painter at the start of a root frame: whether to capture it.
pub(crate) fn take_armed() -> bool {
    std::mem::take(&mut state().armed)
}

/// The painter's hand-off of one captured frame.
pub(crate) fn deliver(frame: Frame<'_>) {
    let mut s = state();
    s.captured += 1;
    if let Some(sink) = s.sink.as_mut() {
        sink(frame);
    }
}

/// Tests: what the painter does with an armed frame, without a GPU (take the flag, count it).
#[doc(hidden)]
pub fn test_capture() {
    if take_armed() {
        deliver(Frame { width: 0, height: 0, format: wgpu::TextureFormat::Bgra8Unorm, bytes: &[] });
    }
}

/// The painter's own texture of the capture size and the buffer it is read back through, both
/// kept from frame to frame.
#[cfg(feature = "winit")]
pub(crate) struct Target {
    pub size: (u32, u32),
    pub format: wgpu::TextureFormat,
    pub texture: wgpu::Texture,
    pub msaa: Option<wgpu::TextureView>,
    pub depth: Option<wgpu::TextureView>,
    pub buffer: wgpu::Buffer,
    pub padded_bytes_per_row: u32,
    /// The frame's rows without padding, reused.
    pub rows: Vec<u8>,
}

#[cfg(feature = "winit")]
impl Target {
    pub fn new(
        device: &wgpu::Device,
        size: (u32, u32),
        format: wgpu::TextureFormat,
        msaa_samples: u32,
        depth_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        let extent = wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("framestep_target"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let samples = msaa_samples.max(1);
        let msaa = (samples > 1).then(|| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("framestep_msaa"),
                    size: extent,
                    mip_level_count: 1,
                    sample_count: samples,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        });
        let depth = depth_format.map(|format| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("framestep_depth"),
                    size: extent,
                    mip_level_count: 1,
                    sample_count: samples,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        });
        let padded_bytes_per_row = wgpu::util::align_to(size.0 * 4, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("framestep_readback"),
            size: u64::from(padded_bytes_per_row) * u64::from(size.1),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self { size, format, texture, msaa, depth, buffer, padded_bytes_per_row, rows: Vec::new() }
    }

    /// Copy the drawn texture into the readback buffer.
    pub fn copy(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_texture_to_buffer(
            self.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &self.buffer,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(self.padded_bytes_per_row), rows_per_image: None },
            },
            self.texture.size(),
        );
    }

    /// After the copy was submitted: wait for it and return the frame's rows (no padding).
    pub fn read(&mut self, device: &wgpu::Device) -> Result<&[u8], String> {
        let slice = self.buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| e.to_string())?;
        rx.recv().map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
        let (w, h) = self.size;
        let row = (w * 4) as usize;
        self.rows.clear();
        self.rows.reserve(row * h as usize);
        {
            let mapped = slice.get_mapped_range().map_err(|e| e.to_string())?;
            for padded in mapped.chunks(self.padded_bytes_per_row as usize).take(h as usize) {
                self.rows.extend_from_slice(&padded[..row]);
            }
        }
        self.buffer.unmap();
        Ok(&self.rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arming_is_taken_once_and_frames_are_counted() {
        set_size(Some((8, 4)));
        assert_eq!(size(), Some((8, 4)));
        let seen = std::sync::Arc::new(Mutex::new(Vec::new()));
        let s = seen.clone();
        set_sink(Some(Box::new(move |f: Frame<'_>| s.lock().unwrap().push((f.width, f.height, f.bytes.len())))));
        let before = captured();
        assert!(!take_armed());
        arm();
        assert!(armed());
        assert!(take_armed());
        assert!(!take_armed());
        deliver(Frame { width: 8, height: 4, format: wgpu::TextureFormat::Bgra8Unorm, bytes: &[0; 128] });
        assert_eq!(captured(), before + 1);
        assert_eq!(*seen.lock().unwrap(), vec![(8, 4, 128)]);
        set_sink(None);
        set_size(None);
    }
}
