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
const Weights = @import("Weights.zig");
const Benchmark = @import("utils.zig").Benchmark;

pub fn main(init: process.Init.Minimal) !void {
    const allocator = heap.smp_allocator;
    var io_instance: Io.Threaded = .init(allocator, .{});
    defer io_instance.deinit();
    const io = io_instance.io();

    var total_benchmark: Benchmark = .start("total", io);
    var step_benchmark: Benchmark = .start("parse arguments", io);
    var it = init.args.iterateAllocator(allocator) catch |err| {
        log.err("{}\n", .{err});
        return;
    };
    defer it.deinit();
    _ = it.skip();

    var cli: Cli = Cli.parse(&it) catch |err| {
        log.err("{}\n", .{err});
        return;
    };
    _ = &cli;
    step_benchmark.stop(io, null);

    step_benchmark = .start("open input", io);
    var input_file: Io.File = Io.Dir.openFile(.cwd(), io, cli.input_tensor_path, .{ .mode = .read_only }) catch |err| {
        log.err("{}\n", .{err});
        return;
    };
    defer input_file.close(io);
    step_benchmark.stop(io, null);

    var input_file_reader_buffer: [64 * 1024]u8 = undefined;
    var reader: Io.File.Reader = input_file.reader(io, &input_file_reader_buffer);

    step_benchmark = .start("parse safetensors", io);
    var parsed_safetensor = safetensors.parseSafetensorsStreaming(allocator, &reader.interface) catch |err| {
        log.err("{}\n", .{err});
        return;
    };
    defer parsed_safetensor.deinit();
    parsed_safetensor.sortByRelativeStart();
    step_benchmark.stop(io, null);

    var weigths: Weights = .init();
    defer weigths.deinit(allocator);

    step_benchmark = .start("build tensor map", io);
    weigths.buildTensorMap(allocator, &parsed_safetensor.values) catch |err| {
        log.err("{}\n", .{err});
        return;
    };
    step_benchmark.stop(io, null);

    step_benchmark = .start("associate NVFP4 weights", io);
    weigths.buildNvfp4WeightsIndex(allocator, parsed_safetensor.values.slice().items(.name)) catch |err| {
        log.err("{}\n", .{err});
        return;
    };
    step_benchmark.stop(io, null);

    step_benchmark = .start("build dispatch list", io);
    weigths.buildDispatchList(allocator, parsed_safetensor.values.len) catch |err| {
        log.err("{}\n", .{err});
        return;
    };
    step_benchmark.stop(io, null);

    step_benchmark = .start("build output metadata", io);
    var output_metadata = safetensors.buildMetadata(allocator, parsed_safetensor.values.slice(), weigths.steps) catch |err| {
        log.err("{}", .{err});
        return;
    };
    defer output_metadata.deinit();
    step_benchmark.stop(io, null);

    const output_ends = output_metadata.values.slice().items(.relative_end);
    const output_data_size: u64 = if (output_ends.len == 0) 0 else output_ends[output_ends.len - 1];

    step_benchmark = .start("open output", io);
    var output_file = Io.Dir.createFile(.cwd(), io, cli.output_tensor_path, .{ .truncate = true }) catch |err| {
        log.err("{}", .{err});
        return;
    };
    defer output_file.close(io);
    step_benchmark.stop(io, null);

    var output_file_writer_buffer: [256 * 1024]u8 = undefined;
    var output_file_wrttier = output_file.writerStreaming(io, &output_file_writer_buffer);

    step_benchmark = .start("write output header", io);
    safetensors.writeHeader(allocator, output_metadata.values.slice(), &output_file_wrttier.interface) catch |err| {
        log.err("{}", .{err});
        return;
    };
    step_benchmark.stop(io, null);

    step_benchmark = .start("convert", io);
    quantization.dequantNvfp4(allocator, io, &reader.interface, &output_file_wrttier.interface, parsed_safetensor.values.slice(), &weigths) catch |err| {
        log.err("{}", .{err});
        return;
    };
    step_benchmark.stop(io, output_data_size);
    step_benchmark = .start("flush", io);
    try output_file_wrttier.interface.flush();
    step_benchmark.stop(io, output_data_size);

    const output_file_stat = try output_file.stat(io);
    total_benchmark.stop(io, output_file_stat.size);
}
