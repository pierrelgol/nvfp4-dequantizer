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

pub fn main(init: process.Init.Minimal) !void {
    const allocator = heap.smp_allocator;

    var cli: Cli = Cli.parse(init, allocator) catch |err| {
        log.err("{}\n", .{err});
        return;
    };
    _ = &cli;
}
