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
const quantization = @import("quantization.zig");
pub const Weights = @This();

const packed_suffix = ".weight_packed";
const local_scale_suffix = ".weight_scale";
const global_scale_suffix = ".weight_global_scale";

pub const TensorBasename = []const u8;
pub const TensorIndex = usize;
pub const TensorMap = std.StringArrayHashMapUnmanaged(TensorIndex);

tensors_index: std.StringArrayHashMapUnmanaged(TensorIndex) = .empty,
weigths: std.MultiArrayList(Nvfp4Weight) = .empty,
steps: []Step,

pub fn init() Weights {
    return .{
        .tensors_index = .empty,
        .weigths = .empty,
        .steps = &.{},
    };
}

pub fn deinit(self: *Weights, allocator: mem.Allocator) void {
    defer self.* = undefined;
    self.tensors_index.deinit(allocator);
    self.weigths.deinit(allocator);
    allocator.free(self.steps);
}

pub fn buildTensorMap(self: *Weights, allocator: mem.Allocator, parsed_tensors: *const std.MultiArrayList(safetensors.MetaData)) !void {
    const tensors = parsed_tensors.slice();
    const names: []const []const u8 = tensors.items(.name);

    for (names, 0..) |name, index| {
        const entry = try self.tensors_index.getOrPut(allocator, name);

        if (entry.found_existing == true) {
            return error.DuplicateTensorName;
        } else {
            entry.value_ptr.* = index;
        }
    }
}

pub fn buildNvfp4WeightsIndex(self: *Weights, allocator: mem.Allocator, names: []const []const u8) !void {
    var buffer: [4096]u8 = undefined;
    var fba: std.heap.FixedBufferAllocator = .init(&buffer);

    for (names, 0..) |name, index| {
        defer fba.reset();
        const basename = mem.cutSuffix(u8, name, packed_suffix) orelse continue;

        const local_scale_key = try mem.concat(
            fba.allocator(),
            u8,
            &.{ basename, local_scale_suffix },
        );

        const global_scale_key = try mem.concat(
            fba.allocator(),
            u8,
            &.{ basename, global_scale_suffix },
        );

        const local_scale_index = self.tensors_index.get(local_scale_key) orelse return error.MissingWeightScale;
        const global_scale_index = self.tensors_index.get(global_scale_key) orelse return error.MissingWeightScale;

        try self.weigths.append(allocator, .{
            .basename = basename,
            .quantized_block = index,
            .local_scale = local_scale_index,
            .global_scale = global_scale_index,
        });
    }
}

pub const Step = union(enum) {
    copy,
    cache_local_scale: usize,
    cache_global_scale: usize,
    dequantize: usize,
};

pub fn buildDispatchList(self: *Weights, allocator: mem.Allocator, tensor_count: usize) !void {
    const nvfp4_weights = self.weigths.slice();
    const quantized_blocks = nvfp4_weights.items(.quantized_block);
    const local_scales = nvfp4_weights.items(.local_scale);
    const global_scales = nvfp4_weights.items(.global_scale);

    const steps = try allocator.alloc(Step, tensor_count);
    errdefer allocator.free(steps);
    @memset(steps, .copy);

    for (local_scales, 0..) |tensor_index, weight_index| {
        try setStepAtIndex(
            steps,
            tensor_index,
            .{
                .cache_local_scale = weight_index,
            },
        );
    }

    for (global_scales, 0..) |tensor_index, weight_index| {
        try setStepAtIndex(
            steps,
            tensor_index,
            .{
                .cache_global_scale = weight_index,
            },
        );
    }

    for (quantized_blocks, 0..) |tensor_index, weight_index| {
        try setStepAtIndex(
            steps,
            tensor_index,
            .{
                .dequantize = weight_index,
            },
        );
    }
    self.steps = steps;
}

fn setStepAtIndex(steps: []Step, index: usize, step: Step) !void {
    std.debug.assert(index < steps.len);

    return switch (steps[index]) {
        .copy => steps[index] = step,
        else => unreachable, // TODO ensure this never exist,
    };
}

pub const Nvfp4Weight = struct {
    basename: TensorBasename,
    quantized_block: TensorIndex,
    local_scale: TensorIndex,
    global_scale: TensorIndex,
};
