const std = @import("std");
const Io = std.Io;
const mem = std.mem;
const heap = std.heap;
const log = std.log;
const process = std.process;

const cli = @import("cli.zig");
const utils = @import("utils.zig");
const safetensors = @import("safetensors.zig");
const quantization = @import("quantization.zig");

pub const input_reader_buffer_size: usize = 64 * 1024;
pub const output_writer_buffer_size: usize = 64 * 1024;

pub fn main(init: std.process.Init.Minimal) !void {
    const gpa = heap.smp_allocator;
    var io_instance = Io.Threaded.init(gpa, .{});
    defer io_instance.deinit();
    const io = io_instance.io();

    var args = init.args.iterateAllocator(gpa) catch |err| {
        return log.err("{}", .{err});
    };
    defer args.deinit();
    _ = args.skip();

    const parsed_args: cli.Result = cli.parseArgs(&args) catch |err| {
        return log.err("{}", .{err});
    };

    const input_file: Io.File = Io.Dir.openFile(.cwd(), io, parsed_args.input_path, .{ .mode = .read_only }) catch |err| {
        return log.err("{}", .{err});
    };
    defer input_file.close(io);

    const output_file: Io.File = Io.Dir.createFile(.cwd(), io, parsed_args.output_path, .{}) catch |err| {
        return log.err("{}", .{err});
    };
    defer output_file.close(io);

    var input_file_reader_buffer: [input_reader_buffer_size]u8 = undefined;
    var input_file_reader: Io.File.Reader = .init(input_file, io, &input_file_reader_buffer);
    const reader: *Io.Reader = &input_file_reader.interface;

    var output_file_writer_buffer: [output_writer_buffer_size]u8 = undefined;
    var output_file_writer: Io.File.Writer = .init(
        output_file,
        io,
        &output_file_writer_buffer,
    );
    const writer: *Io.Writer = &output_file_writer.interface;

    var p0 = utils.Timer.start("safetensor parsing", io);
    var parsed_tensors = try safetensors.parse(gpa, reader);
    defer parsed_tensors.deinit();
    p0.stop(io, parsed_tensors.tensors_start_offset);

    var p1 = utils.Timer.start("build output header", io);
    var output_header = try safetensors.buildDequantizedHeader(
        gpa,
        &parsed_tensors.header,
    );
    defer output_header.deinit();
    p1.stop(io, null);

    const output_infos = output_header.header.tensors.items(.info);
    const output_bytes: u64 = if (output_infos.len == 0)
        0
    else
        output_infos[output_infos.len - 1].data_offsets[1];

    var p2 = utils.Timer.start("write output header", io);
    const header_bytes = try safetensors.writeHeader(
        gpa,
        &output_header.header,
        writer,
    );
    try writer.flush();
    p2.stop(io, header_bytes);

    var p3 = utils.Timer.start("dequantize", io);
    var dequantizer: quantization.Dequantizer = .init(
        gpa,
        io,
        reader,
        writer,
        &parsed_tensors,
    );
    try dequantizer.dequantize(.nvfp4, .f32);
    p3.stop(io, output_bytes);
}
