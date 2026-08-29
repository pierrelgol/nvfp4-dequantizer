const std = @import("std");
const mem = std.mem;
const heap = std.heap;
const Io = std.Io;
const math = std.math;
const json = std.json;
const ascii = std.ascii;
const process = std.process;
const quantization = @import("quantization.zig");
const log = std.log;
pub const Cli = @This();

input_tensor_path: []const u8 = "",
input_tensor_format: quantization.Format = .none,
output_tensor_path: []const u8 = "",
output_tensor_format: quantization.Format = .none,
metrics: bool = false,

pub const empty: Cli = .{};

pub const Error = error{
    MissingInputFileFormat,
    MissingInputFilePath,
    InvalidFormatValue,
    UnknownArgument,
} || mem.Allocator.Error;

pub fn parse(init: process.Init.Minimal, allocator: mem.Allocator) Error!Cli {
    var self: Cli = .empty;

    var it = try init.args.iterateAllocator(allocator);
    defer it.deinit();
    _ = it.skip();

    while (it.next()) |argument| {
        const trimmed = mem.trim(u8, argument, &ascii.whitespace);

        if (mem.eql(u8, "--input", trimmed)) {
            self.input_tensor_path = it.next() orelse return error.MissingInputFilePath;
        } else if (mem.eql(u8, "--input-fmt", trimmed)) {
            const maybe_valid_fmt = it.next() orelse return error.MissingInputFileFormat;
            self.input_tensor_format = quantization.Format.formatFromString(maybe_valid_fmt) orelse return error.InvalidFormatValue;
        } else if (mem.eql(u8, "--output", trimmed)) {
            self.output_tensor_path = it.next() orelse return error.MissingInputFilePath;
        } else if (mem.eql(u8, "--output-fmt", trimmed)) {
            const maybe_valid_fmt = it.next() orelse return error.MissingInputFileFormat;
            self.output_tensor_format = quantization.Format.formatFromString(maybe_valid_fmt) orelse return error.InvalidFormatValue;
        } else {
            log.debug("'{s}'", .{trimmed});
            return error.UnknownArgument;
        }
    }

    return self;
}

pub fn format(
    self: @This(),
    writer: *std.Io.Writer,
) std.Io.Writer.Error!void {
    try json.Stringify.value(self, .{ .whitespace = .indent_4 }, writer);
}
