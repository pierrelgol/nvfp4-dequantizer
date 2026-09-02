const std = @import("std");
const json = std.json;
const mem = std.mem;
const heap = std.heap;
const safetensors = @import("safetensors.zig");
const Io = std.Io;
const Tensor = @import("Tensor.zig");
pub const quantization = @This();

// would be cool to build an interface similar to std.Io
// where a dequantizer/quantizer are just opaque + Vtable
pub const Dequantizer = struct {
    allocator: mem.Allocator = undefined,
    io: std.Io = undefined,
    input: *Io.Reader = undefined,
    output: *Io.Writer = undefined,
    ctx: *const safetensors.ParsedHeader = undefined,

    pub fn init(allocator: mem.Allocator, io: std.Io, input: *Io.Reader, output: *Io.Writer, ctx: *const safetensors.ParsedHeader) Dequantizer {
        return .{
            .allocator = allocator,
            .io = io,
            .input = input,
            .output = output,
            .ctx = ctx,
        };
    }

    fn sequentialInputReaderWorker() void {}
    fn concurrentTensorDequantizerWorker() void {}
    fn sequentialOutputWriterWorker() void {}

    pub fn dequantize(self: *Dequantizer, from: quantization.Scheme, to: quantization.Scheme) void {
        _ = self;
        _ = from;
        _ = to;
    }
};

pub const Scheme = enum {
    nvfp4,
    f32,
};
