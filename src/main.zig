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

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    _ = io;

    var it = init.minimal.args.iterateAllocator(allocator) catch |err| {
        log.err("Fatal : {}", .{err});
        return;
    };
    _ = it.skip();

    var args = Args.parseArgs(&it) catch |err| {
        log.err("Fatal : {}", .{err});
        return;
    };
    _ = &args;

    std.debug.print("{f}", .{args});
}
