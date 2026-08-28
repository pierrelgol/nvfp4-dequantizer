const std = @import("std");
const mem = std.mem;
const heap = std.heap;
const math = std.math;
const log = std.log;
const ascii = std.ascii;
const Io = std.Io;
const process = std.process;
pub const Args = @This();

input: []const u8 = undefined,
output: []const u8 = undefined,
fmt: []const u8 = undefined,

pub const ParseError = error{
    MissingInput,
    MissingOutput,
    MissingFormat,
    InvalidArgument,
};

pub fn parseArgs(it: *process.Args.Iterator) ParseError!Args {
    var args: Args = .{};

    while (it.next()) |argument| {
        const trimmed = mem.trim(u8, argument, &ascii.whitespace);

        if (cmp("-i", trimmed)) {
            args.input = it.next() orelse return error.MissingInput;
        } else if (cmp("-o", trimmed)) {
            args.output = it.next() orelse return error.MissingOutput;
        } else if (cmp("-f", trimmed)) {
            args.fmt = it.next() orelse return error.MissingFormat;
        } else {
            return error.InvalidArgument;
        }
    }

    return args;
}

pub fn cmp(s1: []const u8, s2: []const u8) bool {
    return mem.eql(u8, s1, s2);
}

pub fn format(
    self: @This(),
    writer: *std.Io.Writer,
) std.Io.Writer.Error!void {
    try std.json.Stringify.value(self, .{ .whitespace = .indent_tab }, writer);
}
