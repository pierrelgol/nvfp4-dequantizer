const std = @import("std");
const mem = std.mem;
const heap = std.heap;
const math = std.math;
const fmt = std.fmt;
const log = std.log;
const ascii = std.ascii;
const Io = std.Io;
const process = std.process;
const builtin = @import("builtin");

const Safetensors = @import("Safetensors.zig");

pub const TensorBuilder = @This();

tensor_metadata: []Safetensors.TensorMetaData,
allocator: mem.Allocator,
output: *Io.Queue(QuantizedWeight),

pub fn init(
    allocator: mem.Allocator,
    tensor_metadata: []Safetensors.TensorMetaData,
    output: *Io.Queue(QuantizedWeight),
) TensorBuilder {
    return .{
        .tensor_metadata = tensor_metadata,
        .allocator = allocator,
        .output = output,
    };
}

pub fn deinit(self: *TensorBuilder) void {
    defer self.* = undefined;
}

pub fn run(self: *TensorBuilder, io: std.Io) !void {
    defer self.output.close(io);

    for (self.tensor_metadata, 0..) |w, i| {
        const weight_boundary_start = ".weight_packed";
        const basename_of_current_weight = weightName(w.name, '.') orelse continue;

        if (mem.endsWith(u8, w.name, weight_boundary_start) != true) {
            continue;
        } else {
            // TODO try to see if it wouldn't be better to orefactor findWholeQuantizedWeights
            // to just return a list of all the elements matching base instead of calling it
            // 3 times. bc that's O(N * M * 3) maybe a prefix map ?
            try self.output.putOne(io, .{
                .id = i,
                .name = w.name,
                .weights = w,
                .scale = findWholeQuantizedWeights(self.tensor_metadata, basename_of_current_weight, ".weight_scale"),
                .global_scale = findWholeQuantizedWeights(self.tensor_metadata, basename_of_current_weight, ".weight_global_scale"),
                .global_input_scale = findWholeQuantizedWeights(self.tensor_metadata, basename_of_current_weight, ".input_global_scale"),
            });
        }
    }
}

fn findWholeQuantizedWeights(tensor_metadata: []Safetensors.TensorMetaData, base: []const u8, suffix: []const u8) ?Safetensors.TensorMetaData {
    // TODO look if all the tensor metadata are always in order in the json or not, since we could have a more efficient for loop if we could just pass
    // tensor_metadata[last_iteration_index..] instead of the whole slice.
    const total_len_of_base_plus_suffix = base.len + suffix.len;
    return for (tensor_metadata) |maybe_result| {
        if (maybe_result.name.len != total_len_of_base_plus_suffix) {
            continue;
        }

        if (!std.mem.startsWith(u8, maybe_result.name, base)) {
            continue;
        }

        if (!std.mem.endsWith(u8, maybe_result.name, suffix)) {
            continue;
        }

        break maybe_result;
    } else null;
}

// fn weightBasename(slice: []const u8, sep: u8) ?[]const u8 {
//     const last_index = mem.findScalarLast(u8, slice, sep) orelse return null;
//     return slice[last_index..];
// }

fn weightName(slice: []const u8, sep: u8) ?[]const u8 {
    const last_index = mem.findScalarLast(u8, slice, sep) orelse return null;
    return slice[0..last_index];
}

/// this is the combination of multiple entries in the json combined into one
/// dequantizable unit of concurrency
pub const QuantizedWeight = struct {
    id: usize,
    name: ?[]const u8 = null,
    weights: ?Safetensors.TensorMetaData = null,
    scale: ?Safetensors.TensorMetaData = null,
    global_scale: ?Safetensors.TensorMetaData = null,
    global_input_scale: ?Safetensors.TensorMetaData = null,

    pub fn isComplete(self: *const QuantizedWeight) bool {
        return if (self.name != null and self.weights != null and self.scale != null and self.global_scale != null and self.global_input_scale != null) true else false;
    }
};
