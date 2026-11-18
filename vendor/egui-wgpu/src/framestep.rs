//! SolidCraft: frame-stepped capture. With a capture size set, the root window's frames are drawn
//! into a texture of that size instead of the window's surface, independent of the window (its
//! size, whether it is shown at all); only frames the app arms are drawn and read back, each
//! handed to the sink in order. Frames in between are not drawn: their texture changes wait for
//! the next drawn frame, as they do for a frame without a surface texture.
//!
//! The app arms one frame at a time ([`arm`]) before the frame runs. The painter draws it, copies
//! it into one of a few readback buffers and maps that buffer without waiting: the GPU finishes
//! the frame while the app runs the next one ([`in_flight`] frames at most). Frames are handed on
//! ([`captured`] counts them) in order, as their buffers map; a frame that is not armed waits for
//! the frames in flight. [`Readout::Nv12`] converts the frame on the GPU (BT.709, limited range,
//! 4:2:0, the layout video encoders take), so 12 MB a 4K frame come back instead of 33 MB.

use std::sync::Mutex;
use std::time::{Duration, Instant};

/// The bytes of a frame handed on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pixels {
    /// Rows of `width` pixels, 4 bytes each, blue first.
    Bgra,
    /// Rows of `width` pixels, 4 bytes each, red first.
    Rgba,
    /// `width × height` luma bytes, then `height / 2` rows of `width / 2` (Cb, Cr) pairs:
    /// BT.709, limited range (Y 16–235, Cb/Cr 16–240), chroma the mean of each 2 × 2 block.
    Nv12,
}

impl Pixels {
    /// ffmpeg's name for the layout (`-pix_fmt`).
    pub fn ffmpeg_name(self) -> &'static str {
        match self {
            Self::Bgra => "bgra",
            Self::Rgba => "rgba",
            Self::Nv12 => "nv12",
        }
    }

    /// Bytes of a frame of this size.
    pub fn frame_len(self, width: u32, height: u32) -> usize {
        let px = width as usize * height as usize;
        match self {
            Self::Bgra | Self::Rgba => px * 4,
            Self::Nv12 => px + px / 2,
        }
    }
}

/// What armed frames are read back as.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Readout {
    /// The texture's own 4-byte pixels ([`Pixels::Bgra`] or [`Pixels::Rgba`]).
    #[default]
    Texture,
    /// [`Pixels::Nv12`], converted on the GPU (needs a width divisible by 4; else `Texture`).
    Nv12,
}

/// One frame read back, in `pixels`' layout without row padding, as the window would have shown
/// it; `seq` is the frame's number in arming order (from 0).
pub struct Frame<'a> {
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
    pub pixels: Pixels,
    pub seq: u64,
    pub bytes: &'a [u8],
}

/// Where captured frames go. Called on the window thread, in frame order.
pub type Sink = Box<dyn FnMut(Frame<'_>) + Send>;

/// Frames in flight (armed and not yet handed on) by default: one being read back while the next
/// is drawn.
pub const DEFAULT_IN_FLIGHT: u32 = 2;
/// Readback buffers kept: the most frames in flight.
#[cfg(feature = "winit")]
const SLOTS: usize = 3;

struct State {
    size: Option<(u32, u32)>,
    armed: bool,
    /// Frames armed and taken by the painter so far (the next one's `seq`).
    armed_total: u64,
    captured: u64,
    sink: Option<Sink>,
    readout: Readout,
    in_flight: u32,
    /// When the last frame was armed (profiling).
    armed_at: Option<Instant>,
}

static STATE: Mutex<State> = Mutex::new(State {
    size: None,
    armed: false,
    armed_total: 0,
    captured: 0,
    sink: None,
    readout: Readout::Texture,
    in_flight: DEFAULT_IN_FLIGHT,
    armed_at: None,
});

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

/// What the next armed frames are read back as.
pub fn set_readout(readout: Readout) {
    state().readout = readout;
}

/// What armed frames are read back as.
pub fn readout() -> Readout {
    state().readout
}

/// How many armed frames may be in flight at once, 1 to 3 (1: each is read back before the
/// next frame runs).
pub fn set_in_flight(n: u32) {
    state().in_flight = n.clamp(1, 3);
}

/// How many armed frames may be in flight at once.
pub fn in_flight() -> u32 {
    state().in_flight
}

/// Draw and capture the next frame of the root window.
pub fn arm() {
    let now = Instant::now();
    let mut s = state();
    s.armed = true;
    let prev = s.armed_at.replace(now);
    drop(s);
    if let Some(prev) = prev {
        profile_add("frame_interval", now.saturating_duration_since(prev));
    }
}

/// Whether a frame is armed and not drawn yet.
pub fn armed() -> bool {
    state().armed
}

/// Frames captured (handed on) so far.
pub fn captured() -> u64 {
    state().captured
}

/// Frames armed and taken by the painter so far.
pub fn armed_total() -> u64 {
    state().armed_total
}

/// Where the next captured frames go (`None`: nowhere; they are still counted).
pub fn set_sink(sink: Option<Sink>) {
    state().sink = sink;
}

/// Taken by the painter at the start of a root frame: the armed frame's number, if one is armed.
pub(crate) fn take_armed() -> Option<u64> {
    let mut s = state();
    if !std::mem::take(&mut s.armed) {
        return None;
    }
    let seq = s.armed_total;
    s.armed_total += 1;
    let since = s.armed_at;
    drop(s);
    if let Some(t) = since {
        profile_add("app_frame", t.elapsed());
    }
    Some(seq)
}

/// The painter's hand-off of one captured frame.
pub(crate) fn deliver(frame: Frame<'_>) {
    let t = Instant::now();
    let mut s = state();
    s.captured += 1;
    if let Some(sink) = s.sink.as_mut() {
        sink(frame);
    }
    drop(s);
    profile_add("deliver", t.elapsed());
}

/// Tests: what the painter does with an armed frame, without a GPU (take the flag, count it).
#[doc(hidden)]
pub fn test_capture() {
    if take_armed().is_some() {
        test_deliver();
    }
}

/// Tests: the painter takes the armed frame (it is then in flight) without handing it on.
#[doc(hidden)]
pub fn test_take() -> bool {
    take_armed().is_some()
}

/// Tests: the oldest frame in flight is handed on.
#[doc(hidden)]
pub fn test_deliver() {
    deliver(Frame { width: 0, height: 0, format: wgpu::TextureFormat::Bgra8Unorm, pixels: Pixels::Bgra, seq: 0, bytes: &[] });
}

// ---------------------------------------------------------------- profiling

/// Time spent in one part of making frames: how often, in all, the longest.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stat {
    pub n: u64,
    pub total: Duration,
    pub max: Duration,
}

static PROFILE: Mutex<Vec<(&'static str, Stat)>> = Mutex::new(Vec::new());

/// Count `d` against `name` (painter, app and writer thread alike).
pub fn profile_add(name: &'static str, d: Duration) {
    let mut p = PROFILE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let i = match p.iter().position(|(n, _)| *n == name) {
        Some(i) => i,
        None => {
            p.push((name, Stat::default()));
            p.len() - 1
        }
    };
    let s = &mut p[i].1;
    s.n += 1;
    s.total += d;
    s.max = s.max.max(d);
}

/// The times counted since the last call, in the order first seen; counting starts afresh.
pub fn profile_take() -> Vec<(&'static str, Stat)> {
    std::mem::take(&mut *PROFILE.lock().unwrap_or_else(std::sync::PoisonError::into_inner))
}

// ---------------------------------------------------------------- BT.709 NV12

/// BT.709 luma weights.
const KR: f32 = 0.2126;
const KB: f32 = 0.0722;
const KG: f32 = 1.0 - KR - KB;

fn q8(v: f32) -> u8 {
    (v + 0.5).floor().clamp(0.0, 255.0) as u8
}

/// Limited-range BT.709 (Y, Cb, Cr) of a gamma-encoded RGB colour (0–1 each).
pub fn ycbcr(r: f32, g: f32, b: f32) -> (u8, u8, u8) {
    let l = KR * r + KG * g + KB * b;
    (
        q8(16.0 + 219.0 * l),
        q8(128.0 + 224.0 * (b - l) / (2.0 * (1.0 - KB))),
        q8(128.0 + 224.0 * (r - l) / (2.0 * (1.0 - KR))),
    )
}

/// The CPU's NV12 of 4-byte pixels (`bgra`: blue first, else red first), the same as the GPU's
/// ([`Readout::Nv12`]): for a frame read back as pixels while a recording takes NV12. `width` and
/// `height` must be even; `out` is cleared and filled.
pub fn to_nv12(src: &[u8], width: u32, height: u32, bgra: bool, out: &mut Vec<u8>) {
    let (w, h) = (width as usize, height as usize);
    out.clear();
    out.resize(Pixels::Nv12.frame_len(width, height), 0);
    let rgb = |x: usize, y: usize| {
        let p = &src[(y * w + x) * 4..][..4];
        let (r, g, b) = if bgra { (p[2], p[1], p[0]) } else { (p[0], p[1], p[2]) };
        [f32::from(r) / 255.0, f32::from(g) / 255.0, f32::from(b) / 255.0]
    };
    let (luma, chroma) = out.split_at_mut(w * h);
    for y in 0..h {
        for x in 0..w {
            let [r, g, b] = rgb(x, y);
            luma[y * w + x] = ycbcr(r, g, b).0;
        }
    }
    for y in (0..h).step_by(2) {
        for x in (0..w).step_by(2) {
            let mut sum = [0.0f32; 3];
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let c = rgb(x + dx, y + dy);
                for k in 0..3 {
                    sum[k] += c[k];
                }
            }
            let (_, cb, cr) = ycbcr(sum[0] / 4.0, sum[1] / 4.0, sum[2] / 4.0);
            let i = (y / 2) * w + x;
            chroma[i] = cb;
            chroma[i + 1] = cr;
        }
    }
}

/// The GPU's conversion: one invocation makes 4 × 2 pixels' luma and their two Cb/Cr pairs, as
/// whole words. The texture's values are gamma-encoded unless `srgb` (an sRGB format decodes them
/// on load; they are encoded again here).
const NV12_WGSL: &str = r"
struct Params { width: u32, height: u32, srgb: u32, pad: u32 };
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn rgb(x: u32, y: u32) -> vec3<f32> {
    let c = clamp(textureLoad(src, vec2<i32>(i32(x), i32(y)), 0).rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    if (params.srgb == 0u) { return c; }
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}
fn q8(v: f32) -> u32 { return u32(clamp(floor(v + 0.5), 0.0, 255.0)); }
fn luma(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)); }
fn y8(c: vec3<f32>) -> u32 { return q8(16.0 + 219.0 * luma(c)); }
fn cbcr(c: vec3<f32>) -> u32 {
    let l = luma(c);
    let cb = q8(128.0 + 224.0 * (c.b - l) / 1.8556);
    let cr = q8(128.0 + 224.0 * (c.r - l) / 1.5748);
    return cb | (cr << 8u);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x0 = id.x * 4u;
    let y0 = id.y * 2u;
    let w = params.width;
    if (x0 >= w || y0 >= params.height) { return; }
    for (var row = 0u; row < 2u; row++) {
        let y = y0 + row;
        let word = y8(rgb(x0, y)) | (y8(rgb(x0 + 1u, y)) << 8u) | (y8(rgb(x0 + 2u, y)) << 16u) | (y8(rgb(x0 + 3u, y)) << 24u);
        dst[(y * w + x0) / 4u] = word;
    }
    var uv = 0u;
    for (var k = 0u; k < 2u; k++) {
        let x = x0 + 2u * k;
        let c = (rgb(x, y0) + rgb(x + 1u, y0) + rgb(x, y0 + 1u) + rgb(x + 1u, y0 + 1u)) * 0.25;
        uv = uv | (cbcr(c) << (16u * k));
    }
    dst[(w * params.height + (y0 / 2u) * w + x0) / 4u] = uv;
}
";

/// The GPU's NV12 conversion of one texture into a storage buffer.
#[cfg_attr(not(feature = "winit"), allow(dead_code))]
pub(crate) struct Nv12Pass {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    output: wgpu::Buffer,
    groups: (u32, u32),
}

#[cfg_attr(not(feature = "winit"), allow(dead_code))]
impl Nv12Pass {
    /// For `texture` (a width divisible by 4, an even height, `TEXTURE_BINDING`).
    fn new(device: &wgpu::Device, texture: &wgpu::Texture) -> Self {
        let (w, h) = (texture.width(), texture.height());
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("framestep_nv12"),
            source: wgpu::ShaderSource::Wgsl(NV12_WGSL.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("framestep_nv12"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("framestep_nv12_out"),
            size: Pixels::Nv12.frame_len(w, h) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let srgb = u32::from(texture.format().is_srgb());
        let params: Vec<u8> = [w, h, srgb, 0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let uniform = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("framestep_nv12_params"),
                contents: &params,
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("framestep_nv12"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: uniform.as_entire_binding() },
            ],
        });
        Self { pipeline, bind_group, output, groups: ((w / 4).div_ceil(8), (h / 2).div_ceil(8)) }
    }

    /// Convert the texture and copy the result into `dst` (at least the frame's length).
    fn encode(&self, encoder: &mut wgpu::CommandEncoder, dst: &wgpu::Buffer) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("framestep_nv12"), timestamp_writes: None });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.groups.0, self.groups.1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.output, 0, dst, 0, self.output.size());
    }
}

/// Whether a frame of this size can be read back as NV12.
pub fn nv12_fits(width: u32, height: u32) -> bool {
    width % 4 == 0 && height % 2 == 0
}

// ---------------------------------------------------------------- readback ring

/// A readback buffer: idle, or holding a frame on its way back.
#[cfg(feature = "winit")]
struct Slot {
    buffer: wgpu::Buffer,
    pixels: Pixels,
    /// Bytes a row in the buffer (padded for 4-byte pixels; `width` for NV12).
    stride: u32,
    /// 0: waiting for the GPU, 1: mapped, 2: mapping failed.
    mapped: std::sync::Arc<std::sync::atomic::AtomicU8>,
    seq: u64,
    submission: Option<wgpu::SubmissionIndex>,
}

/// A readback buffer between [`Target::encode_readback`] and [`Target::submitted`].
#[cfg(feature = "winit")]
pub(crate) struct SlotTicket(Slot);

/// The painter's own texture of the capture size and the buffers it is read back through, all
/// kept from frame to frame.
#[cfg(feature = "winit")]
pub(crate) struct Target {
    pub size: (u32, u32),
    pub format: wgpu::TextureFormat,
    pub texture: wgpu::Texture,
    pub msaa: Option<wgpu::TextureView>,
    pub depth: Option<wgpu::TextureView>,
    nv12: Option<Nv12Pass>,
    /// Idle buffers, and buffers in flight (oldest first).
    idle: Vec<Slot>,
    flight: std::collections::VecDeque<Slot>,
    /// A frame's rows without padding, reused.
    rows: Vec<u8>,
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
        let attachment = |label, format| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: extent,
                    mip_level_count: 1,
                    sample_count: samples,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let msaa = (samples > 1).then(|| attachment("framestep_msaa", format));
        let depth = depth_format.map(|f| attachment("framestep_depth", f));
        Self { size, format, texture, msaa, depth, nv12: None, idle: Vec::new(), flight: Default::default(), rows: Vec::new() }
    }

    /// The 4-byte layout of the texture's pixels.
    fn texture_pixels(&self) -> Pixels {
        if matches!(self.format, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb) {
            Pixels::Bgra
        } else {
            Pixels::Rgba
        }
    }

    /// Record the copy of the drawn texture (as `readout` asks) into a free readback buffer;
    /// waits for the oldest frame in flight (and hands it on) when none is free. Give the ticket
    /// to [`Self::submitted`] once the encoder is submitted.
    pub fn encode_readback(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, readout: Readout, seq: u64) -> SlotTicket {
        while self.idle.is_empty() && self.flight.len() >= SLOTS {
            self.finish_oldest(device);
        }
        let (w, h) = self.size;
        let pixels = if readout == Readout::Nv12 && nv12_fits(w, h) { Pixels::Nv12 } else { self.texture_pixels() };
        let stride = match pixels {
            Pixels::Nv12 => w,
            _ => wgpu::util::align_to(w * 4, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
        };
        let need = match pixels {
            Pixels::Nv12 => Pixels::Nv12.frame_len(w, h) as u64,
            _ => u64::from(stride) * u64::from(h),
        };
        let mut slot = match self.idle.iter().position(|s| s.buffer.size() == need) {
            Some(i) => self.idle.swap_remove(i),
            None => {
                // Buffers of another layout are not kept (a recording's layout seldom changes).
                self.idle.clear();
                Slot {
                    buffer: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("framestep_readback"),
                        size: need,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    }),
                    pixels,
                    stride,
                    mapped: Default::default(),
                    seq,
                    submission: None,
                }
            }
        };
        slot.pixels = pixels;
        slot.stride = stride;
        slot.seq = seq;
        if pixels == Pixels::Nv12 {
            self.nv12.get_or_insert_with(|| Nv12Pass::new(device, &self.texture)).encode(encoder, &slot.buffer);
        } else {
            encoder.copy_texture_to_buffer(
                self.texture.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &slot.buffer,
                    layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(stride), rows_per_image: None },
                },
                self.texture.size(),
            );
        }
        SlotTicket(slot)
    }

    /// The encoder holding the copy was submitted: map its buffer (without waiting).
    pub fn submitted(&mut self, ticket: SlotTicket, submission: wgpu::SubmissionIndex) {
        let mut slot = ticket.0;
        let flag = std::sync::Arc::clone(&slot.mapped);
        flag.store(0, std::sync::atomic::Ordering::Release);
        slot.buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            flag.store(if r.is_ok() { 1 } else { 2 }, std::sync::atomic::Ordering::Release);
        });
        slot.submission = Some(submission);
        self.flight.push_back(slot);
    }

    /// Hand on, in order, the frames whose buffers have mapped, then wait for the oldest ones
    /// until at most `left` frames are in flight.
    pub fn harvest_down_to(&mut self, device: &wgpu::Device, left: usize) {
        if self.flight.is_empty() {
            return;
        }
        let _ = device.poll(wgpu::PollType::Poll);
        while let Some(front) = self.flight.front() {
            if self.flight.len() > left || front.mapped.load(std::sync::atomic::Ordering::Acquire) != 0 {
                self.finish_oldest(device);
            } else {
                break;
            }
        }
    }

    /// Wait for the oldest frame in flight and hand it on.
    fn finish_oldest(&mut self, device: &wgpu::Device) {
        let Some(slot) = self.flight.pop_front() else { return };
        if slot.mapped.load(std::sync::atomic::Ordering::Acquire) == 0 {
            let t = Instant::now();
            let poll = match &slot.submission {
                Some(i) => wgpu::PollType::Wait { submission_index: Some(i.clone()), timeout: None },
                None => wgpu::PollType::wait_indefinitely(),
            };
            if let Err(e) = device.poll(poll) {
                log::error!("framestep: waiting for a frame: {e}");
            }
            profile_add("readback_wait", t.elapsed());
        }
        let (w, h) = self.size;
        let format = self.format;
        if slot.mapped.load(std::sync::atomic::Ordering::Acquire) == 1 {
            let t = Instant::now();
            match slot.buffer.slice(..).get_mapped_range() {
                Ok(mapped) => {
                    let row = match slot.pixels {
                        Pixels::Nv12 => w as usize,
                        _ => w as usize * 4,
                    };
                    let bytes: &[u8] = if slot.stride as usize == row {
                        &mapped[..slot.pixels.frame_len(w, h)]
                    } else {
                        self.rows.clear();
                        for padded in mapped.chunks(slot.stride as usize).take(h as usize) {
                            self.rows.extend_from_slice(&padded[..row]);
                        }
                        &self.rows
                    };
                    profile_add("readback_map", t.elapsed());
                    deliver(Frame { width: w, height: h, format, pixels: slot.pixels, seq: slot.seq, bytes });
                }
                Err(e) => {
                    log::error!("framestep: the frame could not be read back: {e}");
                    state().captured += 1;
                }
            }
        } else {
            log::error!("framestep: frame {} could not be read back", slot.seq);
            // Still counted, so a step waiting for it ends.
            state().captured += 1;
        }
        slot.buffer.unmap();
        self.idle.push(slot);
    }

    /// A screenshot of the drawn texture (its submission is already queued): its 4-byte pixels,
    /// no padding, waiting for the GPU. Frames in flight are handed on first.
    pub fn read_now(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<(Pixels, Vec<u8>), String> {
        self.harvest_down_to(device, 0);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("framestep_shot") });
        let SlotTicket(slot) = self.encode_readback(device, &mut encoder, Readout::Texture, u64::MAX);
        let sub = queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        slot.buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let waited = device.poll(wgpu::PollType::Wait { submission_index: Some(sub), timeout: None }).map_err(|e| e.to_string());
        let mapped = waited.and_then(|_| rx.recv().map_err(|e| e.to_string())?.map_err(|e| e.to_string()));
        let (w, h) = self.size;
        let row = w as usize * 4;
        let mut out = Vec::new();
        let result = mapped.and_then(|()| {
            let view = slot.buffer.slice(..).get_mapped_range().map_err(|e| e.to_string())?;
            out.reserve(row * h as usize);
            for padded in view.chunks(slot.stride as usize).take(h as usize) {
                out.extend_from_slice(&padded[..row]);
            }
            Ok(())
        });
        slot.buffer.unmap();
        let pixels = slot.pixels;
        self.idle.push(slot);
        result.map(|()| (pixels, out))
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
        let armed_before = armed_total();
        assert!(take_armed().is_none());
        arm();
        assert!(armed());
        assert_eq!(take_armed(), Some(armed_before));
        assert!(take_armed().is_none());
        assert_eq!(armed_total(), armed_before + 1);
        deliver(Frame { width: 8, height: 4, format: wgpu::TextureFormat::Bgra8Unorm, pixels: Pixels::Bgra, seq: 0, bytes: &[0; 128] });
        assert_eq!(captured(), before + 1);
        assert_eq!(*seen.lock().unwrap(), vec![(8, 4, 128)]);
        set_sink(None);
        set_size(None);
    }

    #[test]
    fn ycbcr_is_bt709_limited_range() {
        assert_eq!(ycbcr(0.0, 0.0, 0.0), (16, 128, 128));
        assert_eq!(ycbcr(1.0, 1.0, 1.0), (235, 128, 128));
        // The primaries, by the BT.709 limited-range matrix.
        assert_eq!(ycbcr(1.0, 0.0, 0.0), (63, 102, 240));
        assert_eq!(ycbcr(0.0, 1.0, 0.0), (173, 42, 26));
        assert_eq!(ycbcr(0.0, 0.0, 1.0), (32, 240, 118));
    }

    #[test]
    fn nv12_planes_and_chroma_means() {
        // 4 × 2: the left 2 × 2 block white, the right one red (BGRA bytes).
        let (white, red) = ([255u8, 255, 255, 255], [0u8, 0, 255, 255]);
        let row: Vec<u8> = [white, white, red, red].concat();
        let src = [row.clone(), row].concat();
        let mut out = Vec::new();
        to_nv12(&src, 4, 2, true, &mut out);
        assert_eq!(out.len(), 12);
        assert_eq!(&out[..8], &[235, 235, 63, 63, 235, 235, 63, 63]);
        assert_eq!(&out[8..], &[128, 128, 102, 240]);
        // The same pixels red first give the same planes.
        let rgba: Vec<u8> = src.chunks(4).flat_map(|p| [p[2], p[1], p[0], p[3]]).collect();
        let mut out2 = Vec::new();
        to_nv12(&rgba, 4, 2, false, &mut out2);
        assert_eq!(out, out2);
    }

    #[test]
    fn pixel_layouts_have_their_sizes() {
        assert_eq!(Pixels::Nv12.frame_len(3840, 2160), 3840 * 2160 * 3 / 2);
        assert_eq!(Pixels::Bgra.frame_len(3840, 2160), 3840 * 2160 * 4);
        assert!(nv12_fits(3840, 2160));
        assert!(!nv12_fits(3842, 2160));
    }

    fn block_on<F: core::future::Future>(f: F) -> F::Output {
        let mut f = core::pin::pin!(f);
        let mut cx = core::task::Context::from_waker(core::task::Waker::noop());
        loop {
            if let core::task::Poll::Ready(v) = f.as_mut().poll(&mut cx) {
                return v;
            }
            std::thread::yield_now();
        }
    }

    /// The GPU's NV12 matches the CPU's ([`to_nv12`]) within one step of rounding, from a plain
    /// and an sRGB texture. Skipped where there is no GPU adapter.
    #[test]
    fn gpu_nv12_matches_the_cpu() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Ok(adapter) = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())) else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let Ok((device, queue)) = block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())) else {
            eprintln!("no GPU device: skipped");
            return;
        };
        let (w, h) = (64u32, 18u32);
        let src: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                let (x, y) = (i % w, i / w);
                [(x * 4) as u8, (y * 14) as u8, ((x * 7 + y * 3) % 256) as u8, 255]
            })
            .collect();
        for format in [wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureFormat::Rgba8UnormSrgb] {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                texture.as_image_copy(),
                &src,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: None },
                texture.size(),
            );
            let pass = Nv12Pass::new(&device, &texture);
            let len = Pixels::Nv12.frame_len(w, h) as u64;
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: len,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            pass.encode(&mut encoder, &readback);
            queue.submit([encoder.finish()]);
            readback.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let gpu = readback.slice(..).get_mapped_range().unwrap().to_vec();
            let mut cpu = Vec::new();
            to_nv12(&src, w, h, false, &mut cpu);
            let worst = gpu.iter().zip(&cpu).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
            assert!(worst <= 1, "{format:?}: GPU and CPU NV12 differ by up to {worst}");
        }
    }
}
