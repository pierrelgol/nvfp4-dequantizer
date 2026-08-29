const std = @import("std");
const mem = std.mem;
const heap = std.heap;
const math = std.math;
const fmt = std.fmt;
const log = std.log;
const ascii = std.ascii;
const Io = std.Io;
const process = std.process;
const Args = @import("Args.zig");
const Safetensors = @import("Safetensors.zig");
const TensorBuilder = @import("TensorBuilder.zig");
const Nvfp4 = @import("Nvfp4.zig");
const builtin = @import("builtin");
pub const Dequantizer = @This();

input: *Io.Queue(TensorBuilder.QuantizedWeight),
output: *Io.Queue(Nvfp4.DecodedWeights),

pub const LoadedQuantizedWeight = struct {
    from: TensorBuilder.QuantizedWeight = undefined,
    scales: []u8 = &.{},
    global_scale: f32 = undefined,
};

pub const WeightKind = union(enum) {
    weights: *LoadedQuantizedWeight,
    weight_scale: *LoadedQuantizedWeight,
    weight_global_scale: *LoadedQuantizedWeight,
};
