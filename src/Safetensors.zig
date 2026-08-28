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
raw_header_arena: heap.ArenaAllocator, // bit of a lazy solution because this means memory usage will max 100MB
output: *Io.Queue(Safetensors.TensorMetaData),

pub const Error = error{
    HeaderTooLarge,
    SomeTingWong, // TODO replace placeholder
};

pub fn init(allocator: mem.Allocator, output: *Io.Queue(Safetensors.TensorMetaData)) Safetensors {
    return .{
        .header_size = undefined,
        .raw_header_arena = .init(allocator),
        .output = output,
    };
}

pub fn deinit(self: *Safetensors) void {
    defer self.* = undefined;
    self.raw_header_arena.deinit();
}

pub fn parseSafetensorHeader(self: *Safetensors, io: std.Io, reader: *Io.Reader) !void {
    const allocator = self.raw_header_arena.allocator();
    var limited_buffer: [std.heap.pageSize()]u8 = undefined;
    var limited = reader.limited(.limited(@intCast(self.header_size)), &limited_buffer);

    var json_reader: std.json.Reader = .init(allocator, &limited.interface);
    defer json_reader.deinit();

    if (try json_reader.next() != .object_begin) {
        return error.SomeTingWong;
    }

    while (true) {
        const entry = try json_reader.nextAlloc(allocator, .alloc_always);

        const name = switch (entry) {
            .allocated_string => |value| value,
            .object_end => break,
            else => return error.SomeTingWong,
        };

        if (mem.eql(u8, "__metadata__", name)) {
            try json_reader.skipValue();
            continue;
        }

        const token = try std.json.parseFromTokenSourceLeaky(RawMetaData, allocator, &json_reader, .{});

        if (token.data_offsets[0] > token.data_offsets[1]) {
            return error.SomeTingWong;
        } else {
            const metadata: TensorMetaData = .{
                .name = name,
                .dtype = token.dtype,
                .shape = token.shape,
                .relative_start = token.data_offsets[0],
                .relative_end = token.data_offsets[1],
            };

            try self.output.putOne(io, metadata);
        }

        if (try json_reader.next() != .end_of_document) {
            return error.SomeTingWong;
        }
    }
}

pub fn decodeJsonHeaderSize(self: *Safetensors, reader: *Io.Reader) !void {
    self.header_size = try reader.takeInt(JsonHeader, std.lang.Endian.little); // TODO look on google wether safetensor is always little endian
    if (self.header_size > (100 * 1024 * 1024)) { // 100MB according to https://www.datacamp.com/blog/safetensors-format
        return error.HeaderTooLarge;
    }
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

pub const RawMetaData = struct {
    dtype: DType,
    shape: []const u64,
    data_offsets: [2]u64,

    pub const View = struct {
        bytes: []align(1) const u8,
    };
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
pub const JsonHeader = u64; // TODO need to find a better name
