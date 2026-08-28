const std = @import("std");
const Safetensors = @import("Safetensors.zig");

const Io = std.Io;
const TensorLoader = @This();

allocator: std.mem.Allocator,
output: *Io.Queue(QuantizedWeights),

pub const QuantizedWeights = struct {
    id: usize,
    input_global_scale: ?Safetensors.TensorMetaData,
    weight: Safetensors.TensorMetaData,
    weight_global_scale: ?Safetensors.TensorMetaData,
    weight_scale: ?Safetensors.TensorMetaData,
};

pub fn init(allocator: std.mem.Allocator, output: *Io.Queue(QuantizedWeights)) TensorLoader {
    return .{
        .allocator = allocator,
        .output = output,
    };
}

pub fn run(self: *TensorLoader, io: Io, tensor_metadata: []Safetensors.TensorMetaData) !void {
    defer self.output.close(io);

    for (tensor_metadata, 0..) |weight, id| {
        const weight_suffix = ".weight_packed";

        if (!std.mem.endsWith(u8, weight.name, weight_suffix)) {
            continue;
        }

        const base = weight.name[0 .. weight.name.len - weight_suffix.len];

        try self.output.putOne(io, .{
            .id = id,
            .weight = weight,
            .weight_scale = findRelated(tensor_metadata, base, ".weight_scale"),
            .weight_global_scale = findRelated(tensor_metadata, base, ".weight_global_scale"),
            .input_global_scale = findRelated(tensor_metadata, base, ".input_global_scale"),
        });
    }
}

fn findRelated(tensor_metadata: []const Safetensors.TensorMetaData, base: []const u8, suffix: []const u8) ?Safetensors.TensorMetaData {
    for (tensor_metadata) |candidate| {
        if (candidate.name.len != base.len + suffix.len) {
            continue;
        }

        if (!std.mem.startsWith(u8, candidate.name, base)) {
            continue;
        }

        if (!std.mem.endsWith(u8, candidate.name, suffix)) {
            continue;
        }

        return candidate;
    }

    return null;
}
