const std = @import("std");
const mem = std.mem;
const heap = std.heap;
const math = std.math;
const fmt = std.fmt;
const log = std.log;
const ascii = std.ascii;
const Io = std.Io;
const process = std.process;
pub const Safetensors = @This();
pub const json = std.json;

header_size: JsonHeader = undefined,
header_json: ?std.json.Parsed(std.json.Value) = null,

pub const Error = error{
    HeaderTooLarge,
};

pub fn init() Safetensors {
    return .{
        .header_size = undefined,
        .header_json = null,
    };
}

pub fn deinit(self: *Safetensors) void {
    defer self.* = undefined;
    if (self.header_json) |some| {
        some.deinit();
    }
}

pub fn loadSafetensors(self: *Safetensors, reader: *Io.Reader, allocator: mem.Allocator) !void {
    try self.decodeJsonHeaderSize(reader);
    try self.decodeJsonHeader(allocator, reader);
}

fn decodeJsonHeaderSize(self: *Safetensors, reader: *Io.Reader) !void {
    self.header_size = try reader.takeInt(JsonHeader, std.lang.Endian.little); // TODO look on google wether safetensor is always little endian
    if (self.header_size > (100 * 1024 * 1024)) { // 100MB according to https://www.datacamp.com/blog/safetensors-format
        return error.HeaderTooLarge;
    }
}

fn decodeJsonHeader(self: *Safetensors, allocator: mem.Allocator, reader: *Io.Reader) !void {
    const json_header_slice = try reader.readAlloc(allocator, self.header_size);
    defer allocator.free(json_header_slice);
    self.header_json = try std.json.parseFromSlice(std.json.Value, allocator, json_header_slice, .{});
}

pub fn format(
    self: @This(),
    writer: *std.Io.Writer,
) std.Io.Writer.Error!void {
    if (self.header_json) |some| {
        try std.json.Stringify.value(some.value, .{ .whitespace = .indent_4 }, writer);
    } else {
        try writer.print("{{(null)}}", .{});
    }
}

pub const DType = enum {
    BOOL,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F4,
    F6_E2M3,
    F6_E3M2,
    F8_E4M3,
    F8_E5M2,
    F8_E8M0,
    F16,
    BF16,
    F32,
    F64,
};

pub const TensorMetaData = struct {
    name: []const u8,
    dtype: DType,
    shape: []const u64,
    relative_start: u64,
    relative_end: u64,

    pub const View = struct {
        bytes: []align(1) const u8,
    };
};

/// 8 bytes according to https://www.datacamp.com/blog/safetensors-format
pub const JsonHeader = u64;
