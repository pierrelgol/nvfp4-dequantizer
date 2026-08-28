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
const TensorLoader = @import("TensorLoader.zig");
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
    const tensor_metadata = try Safetensors.parseSafetensorHeader(arena, reader);

    var job_queue_buffer: [32]TensorLoader.QuantizedWeights = undefined;
    var job_queue: Io.Queue(TensorLoader.QuantizedWeights) = .init(&job_queue_buffer);

    var tensor_loader: TensorLoader = .init(allocator, &job_queue);
    var loader = try io.concurrent(TensorLoader.run, .{ &tensor_loader, io, tensor_metadata });

    while (true) {
        const job = job_queue.getOne(io) catch |err| switch (err) {
            error.Closed => break,
            else => return err,
        };

        try stdout.print(
            "job {d}: {s}, scale={}, global_scale={}, input_scale={}\n",
            .{
                job.id,
                job.weight.name,
                job.weight_scale != null,
                job.weight_global_scale != null,
                job.input_global_scale != null,
            },
        );
    }

    try loader.await(io);
    try stdout.flush();
}
