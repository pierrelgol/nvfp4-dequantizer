const std = @import("std");
const mem = std.mem;
const heap = std.heap;
const math = std.math;
const fmt = std.fmt;
const log = std.log;
const ascii = std.ascii;
const Io = std.Io;
const process = std.process;

pub fn main(init: std.process.Init) !u8 {
    const allocator = init.gpa;
    const io = init.io;
    _ = io;

    var args = init.minimal.args.iterateAllocator(allocator) catch |err| {
        log.err("Fatal : {}", .{err});
        return @intFromError(err);
    };
    _ = args.skip();

    std.debug.print("{s}", .{args.next().?});

    return 0;
}
