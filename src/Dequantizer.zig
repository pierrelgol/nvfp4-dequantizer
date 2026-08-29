const std = @import("std");
const mem = std.mem;
const Io = std.Io;
const Safetensors = @import("Safetensors.zig");
const TensorBuilder = @import("TensorBuilder.zig");
const Nvfp4 = @import("Nvfp4.zig");

pub const Dequantizer = @This();

allocator: mem.Allocator,
io: Io,
reader: *Io.File.Reader,
binary_data_start: u64,
input: *Io.Queue(TensorBuilder.QuantizedWeight),
output: *Io.Queue(DecodedBlock),

pub const DecodedBlock = struct {
    id: usize,
    block_index: usize,
    block_count: usize,
    values: Nvfp4.DecodedWeights,
};

pub const Error = error{
    SomTingWong, // TODO remove placeholder
};

pub fn init(allocator: mem.Allocator, io: Io, reader: *Io.File.Reader, binary_data_start: u64, input: *Io.Queue(TensorBuilder.QuantizedWeight), output: *Io.Queue(DecodedBlock)) Dequantizer {
    return .{
        .allocator = allocator,
        .io = io,
        .reader = reader,
        .binary_data_start = binary_data_start,
        .input = input,
        .output = output,
    };
}

pub fn run(self: *Dequantizer) !void {
    defer self.output.close(self.io);

    while (true) {
        const weight = self.input.getOne(self.io) catch |err| switch (err) {
            error.Closed => break,
            error.Canceled => return err,
        };

        try self.dequantizeWeight(weight);
    }
}

fn dequantizeWeight(self: *Dequantizer, weight: TensorBuilder.QuantizedWeight) !void {
    const quantized_weights_size = try tensorTotalByteSize(weight.weights);
    const quantized_scale_size = try tensorTotalByteSize(weight.scale);
    const quantized_block_size: u64 = @sizeOf(Nvfp4.PackedWeights);
    std.debug.print("quantized_wieght_size {d}\n", .{quantized_weights_size});
    std.debug.print("quantized_scale_size {d}\n", .{quantized_scale_size});
    std.debug.print("quantized_block_size {d}\n", .{quantized_block_size});

    if (quantized_weights_size / quantized_block_size != 0) {
        return error.SomeTingWong;
    }

    const block_count: u64 = quantized_weights_size / quantized_block_size;
    std.debug.print("block_count {d}\n", .{block_count});

    if (block_count != quantized_scale_size) {
        return error.SomTingWong;
    }

    const scaling_factors = try self.allocator.alloc(u8, block_count);
    defer self.allocator.free(scaling_factors);

    try self.readTensor(weight.scale, scaling_factors);
    const global_weight_scale = try self.readGlobalScale(weight.global_scale);

    try self.reader.seekTo(self.fromRelativeToAbsoluteOffset(weight.weights.relative_start));
    for (scaling_factors, 0..) |s, i| {
        var packed_weights: Nvfp4.PackedWeights = undefined;
        try self.reader.interface.readSliceAll(&packed_weights);

        try self.output.putOne(self.io, .{
            .id = weight.id,
            .block_index = i,
            .block_count = block_count,
            .values = Nvfp4.decodePackedWeights(packed_weights, s, global_weight_scale),
        });
    }
}

fn readGlobalScale(self: *Dequantizer, metadata: Safetensors.TensorMetaData) !f32 {
    if (try tensorTotalByteSize(metadata) != @sizeOf(f32)) {
        return error.SomTingWong;
    }

    var out_buffer: [@sizeOf(f32)]u8 = undefined;
    try self.readTensor(metadata, &out_buffer);

    return @bitCast(mem.readInt(u32, &out_buffer, .little)); // TODO also checks if little indian
}

fn readTensor(self: *Dequantizer, metadata: Safetensors.TensorMetaData, out_tensor: []u8) !void {
    const byte_size = try tensorTotalByteSize(metadata);
    std.debug.assert(byte_size == out_tensor.len);

    // if (byte_size != out_tensor.len) {
    //     return error.SomTingWong;
    // }

    try self.reader.seekTo(self.fromRelativeToAbsoluteOffset(metadata.relative_start));
    try self.reader.interface.readSliceAll(out_tensor);
}

fn fromRelativeToAbsoluteOffset(self: *const Dequantizer, relative_offset: u64) u64 {
    return self.binary_data_start + relative_offset;
}

fn tensorTotalByteSize(metadata: Safetensors.TensorMetaData) !u64 {
    if (metadata.relative_end < metadata.relative_start) {
        return error.SomTingWong; // maybe uunreachable ?
    }

    return metadata.relative_end - metadata.relative_start;
}
