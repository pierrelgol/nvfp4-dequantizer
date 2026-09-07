const std = @import("std");
const Io = std.Io;
const Tensor = @import("Tensor.zig");
const json = std.json;
const mem = std.mem;
const safetensors = @This();

pub const Error = error{
    InvalidHeaderSize,
    InvalidHeaderContent,
    InvalidTensorOffsets,
    DuplicateTensorName,
    InacurateHeaderSize,
};

pub const packed_suffix = ".weight_packed";
pub const local_scale_suffix = ".weight_scale";
pub const global_scale_suffix = ".weight_global_scale";
pub const weight_suffix = ".weight";

pub const Header = struct {
    tensors: std.MultiArrayList(Tensor) = .empty,
    tensor_index: std.StringArrayHashMapUnmanaged(Tensor.Index) = .empty,
    metadata: ?Tensor.Metadata = null,

    pub const maximum_header_size = 100 * 1024 * 1024;

    pub fn format(
        self: @This(),
        writer: *std.Io.Writer,
    ) std.Io.Writer.Error!void {
        for (0..self.tensors.len) |i| {
            try writer.print("{f}", .{self.tensors.get(i)});
        }
    }

    fn sortTensorsByDataOffsets(header: *safetensors.Header) void {
        const SortContext = struct {
            slice: std.MultiArrayList(Tensor).Slice,

            pub fn lessThan(ctx: @This(), lhs: usize, rhs: usize) bool {
                return ctx.slice.items(.info)[lhs].data_offsets[0] < ctx.slice.items(.info)[rhs].data_offsets[0];
            }
        };

        header.tensors.sortUnstable(SortContext{
            .slice = header.tensors.slice(),
        });
    }

    fn indexTensorsByName(header: *safetensors.Header, allocator: mem.Allocator) !void {
        try header.tensor_index.ensureTotalCapacity(allocator, header.tensors.len);
        for (header.tensors.items(.name), 0..) |name, index| {
            const gop = try header.tensor_index.getOrPut(allocator, name);

            if (gop.found_existing) {
                return error.DuplicateTensorName;
            } else {
                gop.value_ptr.* = index;
            }
        }
    }
};

pub const ParsedHeader = struct {
    arena: std.heap.ArenaAllocator,
    header: safetensors.Header,
    header_size: u64 = 0,
    tensors_start_offset: u64 = 0,

    pub fn init(allocator: mem.Allocator, header_size: u64) ParsedHeader {
        return .{
            .arena = .init(allocator),
            .header = .{},
            .header_size = header_size,
            .tensors_start_offset = header_size + @sizeOf(u64),
        };
    }

    pub fn deinit(self: *ParsedHeader) void {
        defer self.* = undefined;
        self.arena.deinit();
    }
};

pub fn parse(allocator: mem.Allocator, reader: *Io.Reader) !safetensors.ParsedHeader {
    const header_size = try reader.takeInt(u64, .little);
    return parseJson(allocator, reader, header_size);
}

pub fn parseJson(
    allocator: mem.Allocator,
    reader: *Io.Reader,
    header_size: u64,
) !safetensors.ParsedHeader {
    if (header_size > Header.maximum_header_size) {
        return error.InvalidHeaderSize;
    }

    var result: ParsedHeader = .init(allocator, header_size);
    errdefer result.deinit();
    const arena = result.arena.allocator();

    var limited_reader_buffer: [4096]u8 = undefined;
    var limited_reader: Io.Reader.Limited = .init(
        reader,
        .limited64(header_size),
        &limited_reader_buffer,
    );
    var json_reader: json.Reader = .init(arena, &limited_reader.interface);

    const options: json.ParseOptions = .{
        .allocate = .alloc_always,
        .max_value_len = header_size,
    };

    if (try json_reader.next() != .object_begin) {
        return error.InvalidHeaderContent;
    }

    var sequence: usize = 0;
    while (true) : (sequence += 1) {
        const object = try json_reader.nextAlloc(arena, .alloc_always);

        const object_name = switch (object) {
            .allocated_string => |s| s,
            .object_end => break,
            else => return error.InvalidHeaderContent,
        };

        if (mem.eql(u8, "__metadata__", object_name)) {
            result.header.metadata = try json.innerParse(?Tensor.Metadata, arena, &json_reader, options);
            continue;
        }

        const parsed = try json.innerParse(Tensor.Info, arena, &json_reader, options);

        if (parsed.data_offsets[0] > parsed.data_offsets[1]) {
            return error.InvalidTensorOffsets;
        }

        try result.header.tensors.append(arena, .{
            .info = parsed,
            .name = object_name,
            .sequence = sequence,
        });
    }

    if (limited_reader.remaining.nonzero()) {
        return error.InacurateHeaderSize;
    }

    result.header.sortTensorsByDataOffsets();
    try result.header.indexTensorsByName(arena);

    var end: u64 = 0;
    for (result.header.tensors.items(.info)) |info| {
        if (info.data_offsets[0] < end) return error.InvalidTensorOffsets;
        end = info.data_offsets[1];
    }

    return result;
}

pub fn serializeHeader(allocator: mem.Allocator, stringify_body: []const u8) ![]u8 {
    const padding = (8 - (stringify_body.len % 8)) % 8;
    const header_size = stringify_body.len + padding;

    if (header_size > Header.maximum_header_size) {
        return error.InvalidHeaderSize;
    }

    const bytes = try allocator.alloc(u8, @sizeOf(u64) + header_size);
    mem.writeInt(u64, bytes[0..8], @intCast(header_size), .little);
    @memcpy(bytes[8 .. 8 + stringify_body.len], stringify_body);
    @memset(bytes[8 + stringify_body.len ..], ' ');
    return bytes;
}

// Adapted from huggingface/safetensors safetensors/src/tensor.rs.
test "parses a normal header" {
    const serialized = "\x3c\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    var reader: Io.Reader = .fixed(serialized);

    var parsed = try parse(std.testing.allocator, &reader);
    defer parsed.deinit();

    try std.testing.expectEqual(@as(usize, 1), parsed.header.tensors.len);
    const tensor = parsed.header.tensors.get(0);
    try std.testing.expectEqualStrings("test", tensor.name);
    try std.testing.expectEqual(Tensor.Dtype.I32, tensor.info.dtype);
    try std.testing.expectEqualSlices(u64, &.{ 2, 2 }, tensor.info.shape);
    try std.testing.expectEqual(Tensor.DataOffsets{ 0, 16 }, tensor.info.data_offsets);
}

// Adapted from huggingface/safetensors safetensors/src/tensor.rs.
test "rejects an oversized header" {
    const serialized = "\x3c\x00\x00\x00\x00\xff\xff\xff";
    var reader: Io.Reader = .fixed(serialized);

    try std.testing.expectError(
        error.InvalidHeaderSize,
        parse(std.testing.allocator, &reader),
    );
}

// Adapted from huggingface/safetensors safetensors/src/tensor.rs.
test "accepts a whitespace-padded header" {
    const serialized = "\x06\x00\x00\x00\x00\x00\x00\x00{}\x0d\x20\x09\x0a";
    var reader: Io.Reader = .fixed(serialized);

    var parsed = try parse(std.testing.allocator, &reader);
    defer parsed.deinit();

    try std.testing.expectEqual(@as(usize, 0), parsed.header.tensors.len);
}

// Adapted from huggingface/safetensors safetensors/src/tensor.rs.
test "accepts a zero-sized tensor" {
    const serialized = "\x3b\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,0],\"data_offsets\":[0,0]}}";
    var reader: Io.Reader = .fixed(serialized);

    var parsed = try parse(std.testing.allocator, &reader);
    defer parsed.deinit();

    try std.testing.expectEqual(@as(usize, 1), parsed.header.tensors.len);
    const tensor = parsed.header.tensors.get(0);
    try std.testing.expectEqualStrings("test", tensor.name);
    try std.testing.expectEqual(Tensor.Dtype.I32, tensor.info.dtype);
    try std.testing.expectEqualSlices(u64, &.{ 2, 0 }, tensor.info.shape);
    try std.testing.expectEqual(Tensor.DataOffsets{ 0, 0 }, tensor.info.data_offsets);
}
