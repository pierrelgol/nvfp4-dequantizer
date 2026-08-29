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
const Dequantizer = @import("Dequantizer.zig");
const Nvfp4 = @import("Nvfp4.zig");
const builtin = @import("builtin");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var it = init.minimal.args.iterateAllocator(allocator) catch |err| {
        log.err("Fatal : {}", .{err});
        return;
    };
    _ = it.skip();

    const args = Args.parseArgs(&it) catch |err| {
        log.err("Fatal : {}", .{err});
        return;
    };

    const file = Io.Dir.cwd().openFile(io, args.input, .{}) catch |err| {
        log.err("Fatal : {}", .{err});
        return;
    };
    defer file.close(io);

    var file_reader_buffer: [std.heap.pageSize()]u8 = undefined;
    var file_reader: Io.File.Reader = .init(file, io, &file_reader_buffer);
    const reader: *Io.Reader = &file_reader.interface;

    const arena = init.arena.allocator();
    const parsedTensorMetadata = try Safetensors.parseSafetensorHeader(arena, reader);
    const tensor_metadata = parsedTensorMetadata.tensors;
    std.sort.insertion(Safetensors.TensorMetaData, tensor_metadata, {}, lessThanTensor);

    // TODO: rework this awful code
    const output_metadata = try buildDequantizedHeader(arena, tensor_metadata);
    const output_file = try Io.Dir.cwd().createFile(io, args.output, .{});
    defer output_file.close(io);
    var output_writer_buffer: [64 * 1024]u8 = undefined;
    var output_writer = output_file.writerStreaming(io, &output_writer_buffer);
    const output: *Io.Writer = &output_writer.interface;

    var copy_reader_buffer: [64 * 1024]u8 = undefined;
    var copy_reader: Io.File.Reader = .init(file, io, &copy_reader_buffer);

    try Safetensors.writeHeader(arena, output_metadata, output);

    var tensor_builder_output_queue_buffer: [32]TensorBuilder.QuantizedWeight = undefined;
    var tensor_builder_output_queue: Io.Queue(TensorBuilder.QuantizedWeight) = .init(&tensor_builder_output_queue_buffer);

    var tensor_builder: TensorBuilder = .init(allocator, tensor_metadata, &tensor_builder_output_queue);
    defer tensor_builder.deinit();

    var tensor_builder_future = try io.concurrent(TensorBuilder.run, .{ &tensor_builder, io });
    defer tensor_builder_future.cancel(io) catch {};

    var dequantizer_output_queue_buffer: [32]Dequantizer.DecodedBlock = undefined;
    var dequantizer_output_queue: Io.Queue(Dequantizer.DecodedBlock) = .init(&dequantizer_output_queue_buffer);

    var dequantizer: Dequantizer = .init(
        allocator,
        io,
        &file_reader,
        @intCast(parsedTensorMetadata.positionnal_binary_data_start),
        &tensor_builder_output_queue,
        &dequantizer_output_queue,
    );

    var dequantizer_future = try io.concurrent(Dequantizer.run, .{&dequantizer});
    defer dequantizer_future.cancel(io) catch {};

    for (tensor_metadata, 0..) |metadata, tensor_id| {
        if (isDiscardableAfterDequantization(metadata.name)) continue;

        if (mem.endsWith(u8, metadata.name, ".weight_packed")) {
            const packed_len = metadata.relative_end - metadata.relative_start;
            const block_count: usize = @intCast(packed_len / @sizeOf(Nvfp4.PackedWeights));
            for (0..block_count) |block_index| {
                const decoded = try dequantizer_output_queue.getOne(io);
                if (decoded.id != tensor_id or decoded.block_index != block_index) return error.UnexpectedDecodedBlock;
                try output.writeAll(mem.asBytes(&decoded.values));
            }
        } else {
            try cloneOutTensor(
                &copy_reader,
                output,
                @intCast(parsedTensorMetadata.positionnal_binary_data_start),
                metadata,
            );
        }
    }

    try dequantizer_future.await(io);
    try tensor_builder_future.await(io);
    try output.flush();
}

fn lessThanTensor(_: void, a: Safetensors.TensorMetaData, b: Safetensors.TensorMetaData) bool {
    return a.relative_start < b.relative_start;
}

fn isDiscardableAfterDequantization(name: []const u8) bool {
    // TODO ensure this is how it's done, I think it make sense to discard those
    // since those are only needed for dequantizing the nvfp4

    if (mem.endsWith(u8, name, ".weight_scale")) {
        return true;
    } else if (mem.endsWith(u8, name, ".weight_global_scale")) {
        return true;
    } else if (mem.endsWith(u8, name, ".input_global_scale")) {
        return true;
    } else {
        return false;
    }
}

fn buildDequantizedHeader(allocator: mem.Allocator, input: []const Safetensors.TensorMetaData) ![]Safetensors.TensorMetaData {
    var result: std.ArrayListUnmanaged(Safetensors.TensorMetaData) = .empty;
    var output_offset: u64 = 0;

    for (input) |metadata| {
        if (isDiscardableAfterDequantization(metadata.name)) continue;

        var output_metadata = metadata;
        var byte_len = metadata.relative_end - metadata.relative_start;

        if (mem.endsWith(u8, metadata.name, ".weight_packed")) {
            const suffix = ".weight_packed";
            const base = metadata.name[0 .. metadata.name.len - suffix.len];

            output_metadata.name = try mem.concat(allocator, u8, &.{ base, ".weight" });
            output_metadata.dtype = .F32;

            const shape = try allocator.dupe(u64, metadata.shape);

            shape[shape.len - 1] *= 2;
            output_metadata.shape = shape;
            byte_len *= 2 * @sizeOf(f32);
        }

        output_metadata.relative_start = output_offset;
        output_metadata.relative_end = output_offset + byte_len;
        output_offset = output_metadata.relative_end;

        try result.append(allocator, output_metadata);
    }

    return result.toOwnedSlice(allocator);
}

fn cloneOutTensor(reader: *Io.File.Reader, writer: *Io.Writer, binary_data_start: u64, metadata: Safetensors.TensorMetaData) !void {
    try reader.seekTo(binary_data_start + metadata.relative_start);
    var remaining = metadata.relative_end - metadata.relative_start;
    var buffer: [64 * 1024]u8 = undefined;

    while (remaining != 0) {
        const len: u64 = @intCast(@min(remaining, buffer.len));

        try reader.interface.readSliceAll(buffer[0..len]);
        try writer.writeAll(buffer[0..len]);
        remaining -= len;
    }
}
