const std = @import("std");
const Io = std.Io;
const Tensor = @import("Tensor.zig");
const json = std.json;
const mem = std.mem;
const heap = std.heap;

pub const Header = struct {
    tensors: json.ArrayHashMap(Tensor.Info) = .{},
    metadata: ?json.ArrayHashMap(Header.Metadata) = null,

    pub const Metadata = []const u8;
};

pub fn parse(allocator : mem.Allocator, io : std.Io, reader : *Io.Reader) !void {
    _ = all
}
