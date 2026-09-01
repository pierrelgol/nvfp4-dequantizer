const std = @import("std");
const mem = std.mem;
const heap = std.heap;
const Io = std.Io;
const ascii = std.ascii;
const process = std.process;
const json = std.json;

pub const Result = struct {
    input_path: []const u8 = "",
    output_path: []const u8 = "",

    pub fn format(
        self: @This(),
        writer: *std.Io.Writer,
    ) std.Io.Writer.Error!void {
        try json.Stringify.value(self, .{ .whitespace = .indent_4 }, writer);
    }
};

pub fn parseArgs(it: *process.Args.Iterator) !Result {
    return .{
        .input_path = it.next() orelse return error.MissingInputSafetensorFile,
        .output_path = it.next() orelse return error.MissingOutputSafetensorFile,
    };
}
