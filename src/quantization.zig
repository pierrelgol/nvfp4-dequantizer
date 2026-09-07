const std = @import("std");
const mem = std.mem;
const Io = std.Io;
const math = std.math;
const json = std.json;
const heap = std.heap;
const safetensors = @import("safetensors.zig");
const nvfp4 = @import("nvfp4.zig");
const Tensor = @import("Tensor.zig");

pub const Dequantizer = struct {
    allocator: mem.Allocator,
    io: std.Io,
    input: *Io.Reader,
    interface: Io.Reader,
    maybe_err: ?anyerror = null,
    parsed: safetensors.ParsedHeader,
    steps: []Step = &.{},
    cache: []WeightCache = &.{},
    pool: ?*Pool = null,
    workers: Io.Group = .init,
    workers_started: bool = false,

    const vtable: Io.Reader.VTable = &.{};

    pub fn init() void {}
    pub fn deinit() void {}

    pub fn stream(reader: *Io.Reader, writer: *Io.Writer, limit: Io.Limit) Io.Reader.StreamError!usize {
        _ = reader;
        _ = writer;
        _ = limit;
    }
};

pub const Step = enum {};
pub const WeightCache = struct {};
pub const Pool = struct {};
pub const Block = struct {};
