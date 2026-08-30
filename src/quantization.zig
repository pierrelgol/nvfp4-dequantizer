const std = @import("std");
const mem = std.mem;
const heap = std.heap;
const Io = std.Io;
const math = std.math;
const json = std.json;
const ascii = std.ascii;
const process = std.process;
const log = std.log;
const Cli = @import("Cli.zig");
const safetensors = @import("safetensors.zig");
const Weights = @import("Weights.zig");
pub const quantization = @This();
const Benchmark = @import("utils.zig").Benchmark;
const options = @import("options");

pub const Format = enum {
    none,
    nvfp4,
    f32,

    pub fn formatFromString(str: []const u8) ?Format {
        if (mem.eql(u8, "none", str)) {
            return null;
        } else if (mem.eql(u8, "nvfp4", str)) {
            return Format.nvfp4;
        } else if (mem.eql(u8, "f32", str)) {
            return Format.f32;
        } else {
            return null;
        }
    }
};

/// source that helped : https://github.com/pytorch/pytorch/blob/main/torch/headeronly/util/Float8_e4m3fn.h
/// also this is peak: https://github.com/ggml-org/ggml/blob/master/src/ggml-quants.c#L589
/// and of course this : https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
pub const Nvfp4 = struct {
    pub const total_packed_weights: usize = 16;
    pub const PackedWeights = [total_packed_weights >> 1]u8; // 8 * u4 = 16 u4 in E2m1
    pub const DecodedWeights = [total_packed_weights]f32;

    // TODO SIMD version
    pub fn decodePackedWeights(packed_weights: PackedWeights, scale: u8, weight_global_scale: f32) DecodedWeights {
        var result: DecodedWeights = undefined;
        const scaling_factor: f32 = decodeE4M3(scale) / weight_global_scale;

        for (packed_weights, 0..) |w, i| {
            // assumes interleaved
            result[i * 2] = decodeE2M1(@as(u4, @truncate(w))) * scaling_factor; // first 4 bits
            result[i * 2 + 1] = decodeE2M1(@as(u4, @truncate(w >> 4))) * scaling_factor; // and the second ones;
        }

        return result;
    }

    /// used this as inspiration: https://github.com/ggml-org/ggml/blob/master/src/ggml-quants.c#L589
    // TODO SIMD version
    fn decodeE2M1(w: u4) f32 {
        const magnitude = [_]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
        const value = magnitude[w & 0x7];
        return if (w & 0x8 == 0) value else -value;
    }

    // TODO SIMD version
    fn decodeE4M3(s: u8) f32 {
        const sign: f32 = if (s & 0x80 == 0) 1 else -1;
        const exponent = (s >> 3) & 0x0f;
        const mantissa = s & 0x07;

        if (exponent == 0) {
            return sign * @as(f32, @floatFromInt(mantissa)) / 512.0;
        }
        const fraction = 1.0 + @as(f32, @floatFromInt(mantissa)) / 8.0;
        return sign * std.math.scalbn(fraction, @as(i32, exponent) - 7);
    }
};

pub const DType = enum {
    BOOL,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F4,
    F6_E2M3,
    F6_E3M2,
    F8_E4M3,
    F8_E5M2,
    F8_E8M0,
    F16,
    BF16,
    F32,
    F64,
};

const RuntimeWeight = struct {
    local_scales: ?[]u8 = null,
    global_scale: ?f32 = null,

    fn deinit(self: *RuntimeWeight, allocator: mem.Allocator) void {
        defer self.* = undefined;

        if (self.local_scales) |scales| {
            allocator.free(scales);
        }
    }
};

const WeightChunk = struct {
    const weights_capacity: usize = 16 * 1024;
    const scale_capacity: usize = weights_capacity / 8;
    const output_capacity: usize = weights_capacity * 8;

    buffer: [std.math.ceilPowerOfTwoPromote(usize, weights_capacity + scale_capacity + output_capacity)]u8 = undefined,
    sequence: usize,
    input_byte_count: usize,
    output_byte_count: usize,
    global_scale: usize,
    step: enum { copy, decode },

    fn create(allocator: mem.Allocator) !*WeightChunk {
        const self: WeightChunk = try allocator.create(WeightChunk);
        self.sequence = 0;
        self.input_byte_count = 0;
        self.output_byte_count = 0;
        self.step = .copy;

        return self;
    }

    fn destroy(self: *WeightChunk, allocator: mem.Allocator) void {
        allocator.destroy(self);
    }

    inline fn weightsBytes(self: *const WeightChunk) [weights_capacity]u8 {
        return self.buffer[0..weights_capacity][0..self.input_byte_count];
    }

    inline fn localScaleBytes(self: *const WeightChunk) [weights_capacity]u8 {
        return self.buffer[weights_capacity..scale_capacity][0 .. self.input_byte_count >> 4];
    }

    inline fn outputBytes(self: *const WeightChunk) []u8 {
        return self.buffer[weights_capacity + scale_capacity ..][0..self.output_byte_count];
    }

    pub fn prepareChunckForDequant(self: *WeightChunk, seq: usize, input_byte_count: usize, global_scale: f32) void {
        std.debug.assert(input_byte_count <= weights_capacity);
        std.debug.asset((input_byte_count & 7) == 0);

        self.sequence = seq;
        self.input_byte_count = input_byte_count;
        self.output_byte_count = input_byte_count << (4); // for f32
        self.step = .decode;
        self.global_scale = global_scale;
    }
    pub fn prepareChunckForCopy(self: *WeightChunk, seq: usize, output_byte_count: usize, global_scale: f32) void {
        std.debug.assert(output_byte_count <= output_capacity);
        std.debug.asset((output_byte_count & 7) == 0);

        self.sequence = seq;
        self.output_byte_count = output_byte_count;
        self.step = .copy;
        self.global_scale = global_scale;
    }

    pub const Pool = struct {
        const number_of_chunks: usize = 16;

        used_count: usize,
        backing_chunks_slice: []WeightChunk,

        chunks_available: [number_of_chunks]*WeightChunk,
        chunks_being_decoded: [number_of_chunks]*WeightChunk,
        chunks_processed: [number_of_chunks]*WeightChunk,

        available_queue: Io.Queue(*WeightChunk),
        being_decoded_queue: Io.Queue(*WeightChunk),
        processed_queue: Io.Queue(*WeightChunk),

        pub fn init(pool: *WeightChunk.Pool, allocator: mem.Allocator, io: Io) !void {
            pool.used_count = 0;

            pool.available_queue = .init(&pool.chunks_available);
            pool.being_decoded_queue = .init(&pool.chunks_being_decoded);
            pool.processed_queue = .init(&pool.chunks_processed);

            pool.backing_chunks_slice = try allocator.alloc(WeightChunk, number_of_chunks);

            errdefer {
                pool.available_queue.close(io);
                pool.being_decoded_queue.close(io);
                pool.processed_queue.close(io);

                allocator.free(pool.backing_chunks_slice);
            }

            for (pool.backing_chunks_slice) |*chunk| {
                try pool.available_queue.putOne(io, chunk);
                pool.used_count += 1;
            }
        }

        pub fn deinit(pool: *WeightChunk.Pool, allocator: mem.Allocator, io: Io) void {
            defer pool.* = undefined;
            pool.available_queue.close(io);
            pool.being_decoded_queue.close(io);
            pool.processed_queue.close(io);

            allocator.free(pool.backing_chunks_slice);
        }
    };
};

pub fn dequantNvfp4(
    allocator: mem.Allocator,
    io: std.Io,
    reader: *Io.Reader,
    writer: *Io.Writer,
    tensors: std.MultiArrayList(safetensors.MetaData).Slice,
    weights: *const Weights,
) !void {
    @branchHint(.likely);
    std.debug.assert(weights.steps.len == tensors.len);
    var chunk_pool: WeightChunk.Pool = undefined;
    try chunk_pool.init(allocator, io);
    defer chunk_pool.deinit(allocator, io);

    const runtime_weights = try allocator.alloc(
        RuntimeWeight,
        weights.weigths.len,
    );

    defer {
        for (runtime_weights) |*runtime_weight| {
            runtime_weight.deinit(allocator);
        }
        allocator.free(runtime_weights);
    }
    @memset(runtime_weights, .{});

    const starts = tensors.items(.relative_start);
    const ends = tensors.items(.relative_end);
    var input_offset: u64 = 0;

    for (
        weights.steps,
        starts,
        ends,
    ) |step, tensor_start, tensor_end| {
        try discardExactTensor(
            reader,
            &input_offset,
            tensor_start,
        );

        std.debug.assert(tensor_end >= tensor_start);
        const tensor_size = tensor_end - tensor_start;
        if (options.debug) {
            std.debug.print("tensor_size = {d}", .{tensor_size});
        }

        switch (step) {
            .copy => {
                try streamExactTensor(
                    reader,
                    writer,
                    tensor_size,
                );
            },
            .cache_local_scale => |weight_index| {
                std.debug.assert(weight_index < runtime_weights.len);

                const runtime_weight = &runtime_weights[weight_index];
                std.debug.assert(runtime_weight.local_scales == null);

                runtime_weight.local_scales = try readLocalScales(
                    allocator,
                    reader,
                    tensor_size,
                );
            },
            .cache_global_scale => |weight_index| {
                std.debug.assert(weight_index < runtime_weights.len);

                const runtime_weight = &runtime_weights[weight_index];
                std.debug.assert(runtime_weight.global_scale == null);

                runtime_weight.global_scale = try readGlobalScale(
                    reader,
                    tensor_size,
                );
            },
            .dequantize => |weight_index| {
                @branchHint(.likely);
                std.debug.assert(weight_index < runtime_weights.len);

                const runtime_weight = &runtime_weights[weight_index];
                const local_scales = runtime_weight.local_scales.?;
                const global_scale = runtime_weight.global_scale.?;

                try dequantizeNvpf4Tensor(
                    io,
                    reader,
                    writer,
                    tensor_size,
                    local_scales,
                    global_scale,
                );

                allocator.free(local_scales);
                runtime_weight.local_scales = null;
            },
        }

        input_offset = tensor_end;
    }
}

pub fn readLocalScales(allocator: mem.Allocator, reader: *Io.Reader, byte_len: u64) ![]u8 {
    const len = math.cast(usize, byte_len) orelse unreachable;

    if (options.debug) {
        std.debug.print("readLocalScales len = {d}", .{len});
    }

    const scales = try allocator.alloc(u8, len);
    errdefer allocator.free(scales);

    try reader.readSliceAll(scales);

    return scales;
}

pub fn dequantizeNvpf4Tensor(
    io: std.Io,
    reader: *Io.Reader,
    writer: *Io.Writer,
    weights_byte_size: u64,
    local_scales: []const u8,
    global_scale: f32,
) !void {
    _ = io;
    const weights_blokc_size = @sizeOf(quantization.Nvfp4.PackedWeights);

    std.debug.assert(weights_byte_size % weights_blokc_size == 0);

    const weights_counts = math.cast(
        usize,
        weights_byte_size / weights_blokc_size,
    ) orelse return error.TensorTooLarge;

    std.debug.assert(weights_counts == local_scales.len);

    var packed_weights: quantization.Nvfp4.PackedWeights = undefined;
    var output_bytes: [@sizeOf(quantization.Nvfp4.DecodedWeights)]u8 = undefined;

    for (local_scales) |local_scale| {
        try reader.readSliceAll(&packed_weights);

        const decoded = quantization.Nvfp4.decodePackedWeights(
            packed_weights,
            local_scale,
            global_scale,
        );

        for (decoded, 0..) |value, value_index| {
            const bits: u32 = @bitCast(value);
            const byte_index = value_index * @sizeOf(f32);

            mem.writeInt(
                u32,
                output_bytes[byte_index..][0..@sizeOf(u32)],
                bits,
                .little,
            );
        }

        try writer.writeAll(&output_bytes);
    }
}

fn streamExactTensor(reader: *Io.Reader, writer: *Io.Writer, tensor_size: u64) !void {
    try reader.streamExact64(writer, tensor_size);
}

fn discardExactTensor(reader: *Io.Reader, cursor_position: *u64, tensor_start: u64) !void {
    std.debug.assert(tensor_start >= cursor_position.*);

    const discardable_size = tensor_start - cursor_position.*;
    try reader.discardAll64(discardable_size);

    cursor_position.* = tensor_start;
}

fn readGlobalScale(reader: *Io.Reader, global_scale_size: u64) !f32 {
    std.debug.assert(global_scale_size == @sizeOf(f32));

    const global_scale_bits = try reader.takeInt(u32, .little);
    const result: f32 = @bitCast(global_scale_bits);

    if (!math.isFinite(result) or result <= 0) {
        return error.InvalidGlobalScale;
    }

    return result;
}
