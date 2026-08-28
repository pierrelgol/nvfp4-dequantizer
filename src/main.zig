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

    // const file_stats = file.stat(io) catch |err| {
    //     log.err("Fatal : {}", .{err});
    //     return;
    // };

    var file_reader_buffer: [std.heap.pageSize()]u8 = undefined;
    var file_reader: Io.File.Reader = .init(file, io, &file_reader_buffer);
    const reader: *Io.Reader = &file_reader.interface;

    // var stdout_writer_buffer: [std.heap.pageSize()]u8 = undefined;
    // var stdout_writer: Io.File.Writer = Io.File.stdout().writerStreaming(io, &stdout_writer_buffer);
    // const stdout: *Io.Writer = &stdout_writer.interface;

    var tensor_output_queue_buffer: [32]Safetensors.TensorMetaData = undefined;
    var tensor_output_queue: Io.Queue(Safetensors.TensorMetaData) = .init(&tensor_output_queue_buffer);
    defer tensor_output_queue.close(io);

    var tensor: Safetensors = .init(allocator, &tensor_output_queue);
    defer tensor.deinit();

    try tensor.decodeJsonHeaderSize(reader);
    var future = try io.concurrent(Safetensors.parseSafetensorHeader, .{ &tensor, io, reader });
    defer {
        if (future.cancel(io)) |_| {} else |err| {
            log.err("Fatal : {}\n", .{err});
        }
    }
}
