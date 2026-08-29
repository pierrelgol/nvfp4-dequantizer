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

pub fn main(init: process.Init.Minimal) !void {
    const allocator = heap.smp_allocator;
    var io_instance: Io.Threaded = .init(allocator, .{});
    defer io_instance.deinit();
    const io = io_instance.io();

    var it = init.args.iterateAllocator(allocator) catch |err| {
        log.err("{}\n", .{err});
        return;
    };
    defer it.deinit();

    var cli: Cli = Cli.parse(&it) catch |err| {
        log.err("{}\n", .{err});
        return;
    };
    _ = &cli;

    var input_file: Io.File = Io.Dir.openFile(.cwd(), io, cli.input_tensor_path, .{ .mode = .read_only }) catch |err| {
        log.err("{}\n", .{err});
        return;
    };

    var input_file_reader_buffer: [heap.pageSize()]u8 = undefined;
    var reader: Io.File.Reader = input_file.reader(io, &input_file_reader_buffer);

    var parsed_safetensor = safetensors.parseSafetensorsStreaming(allocator, &reader.interface) catch |err| {
        log.err("{}\n", .{err});
        return;
    };
    defer parsed_safetensor.deinit();
}
