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

    var stdout_writer_buffer: [std.heap.pageSize()]u8 = undefined;
    var stdout_writer: Io.File.Writer = Io.File.stdout().writerStreaming(io, &stdout_writer_buffer);
    const stdout: *Io.Writer = &stdout_writer.interface;

    const arena = init.arena.allocator();
    const parsedTensorMetadata = try Safetensors.parseSafetensorHeader(arena, reader);
    const tensor_metadata = parsedTensorMetadata.tensors;

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

    var decoded_block_count: usize = 0;
    while (true) {
        _ = dequantizer_output_queue.getOne(io) catch |err| switch (err) {
            error.Closed => break,
            error.Canceled => return err,
        };
        decoded_block_count += 1;
    }

    try dequantizer_future.await(io);
    try tensor_builder_future.await(io);

    log.info("total blocks {d} Nfvp4 -> f32", .{decoded_block_count});

    try stdout.flush();
}
