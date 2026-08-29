const std = @import("std");
const mem = std.mem;
const heap = std.heap;
const math = std.math;
const fmt = std.fmt;
const log = std.log;
const ascii = std.ascii;
const Io = std.Io;
const process = std.process;
const json = std.json;
pub const Safetensors = @This();

pub const Error = error{
    HeaderTooLarge,
    SomeTingWong, // TODO replace placeholder
};

// TODO now that I think about it there could be one approach which consist of simply scanning to find
// weights boundarie during safetensors header parse
// because my planned pipeline currently is like
//
// [1T parse] -> [1T merge into units of concurrency] -> [NT consume and dequantize]
//
// but potentially if this is feasible we could have somnething like this
//
// [1T find_boundaries] -> [NT merges boundaries in units of concurrency] -> [NT consume and dequantize]
//

pub const ParsedTensorMetadata = struct {
    tensors: []TensorMetaData,
    positionnal_binary_data_start: usize,
};

pub fn parseSafetensorHeader(allocator: mem.Allocator, reader: *Io.Reader) !ParsedTensorMetadata {
    var result: ParsedTensorMetadata = undefined;

    const header_size = try decodeJsonHeaderSize(reader);
    var list: std.ArrayListUnmanaged(TensorMetaData) = .empty;

    // this is where the relative index will really start
    // because in the format of safetensors the first bytes looks like this
    // [header_size: [0..8]][json_header: [0..100 * 1024 * 1024]]
    result.positionnal_binary_data_start = @sizeOf(u64) + header_size;

    var limited_buffer: [std.heap.pageSize()]u8 = undefined;
    var limited = reader.limited(.limited(@intCast(header_size)), &limited_buffer);

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

        const token = try std.json.innerParse(RawMetaData, allocator, &json_reader, .{
            .allocate = .alloc_always,
            .max_value_len = std.json.default_max_value_len,
        });

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

            try list.append(allocator, metadata);
        }
    }

    if (try json_reader.next() != .end_of_document) {
        return error.SomeTingWong;
    }

    result.tensors = try list.toOwnedSlice(allocator);
    return result;
}

fn decodeJsonHeaderSize(reader: *Io.Reader) !u64 {
    const header_size = try reader.takeInt(JsonHeader, std.lang.Endian.little); // TODO look on google wether safetensor is always little endian

    //TODO verify it's inclusive
    if (header_size > (100 * 1024 * 1024)) { // 100MB according to https://www.datacamp.com/blog/safetensors-format
        return error.HeaderTooLarge;
    }

    return header_size;
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

fn stringifyHeader(tensor_metadata: []const TensorMetaData, writer: *Io.Writer) Io.Writer.Error!void {
    var stringify: std.json.Stringify = .{
        .writer = writer,
        .options = .{ .whitespace = .indent_4 },
    };

    try stringify.beginObject();
    for (tensor_metadata) |metadata| {
        try stringify.objectField(metadata.name);

        const item: RawMetaData = .{
            .dtype = metadata.dtype,
            .shape = metadata.shape,
            .data_offsets = .{ metadata.relative_start, metadata.relative_end },
        };

        try stringify.write(item);
    }
    try stringify.endObject();
}

pub fn writeHeader(allocator: mem.Allocator, tensor_metadata: []const TensorMetaData, writer: *Io.Writer) !void {
    var allocating: Io.Writer.Allocating = .init(allocator);
    defer allocating.deinit();

    try stringifyHeader(tensor_metadata, &allocating.writer);
    const json_bytes = allocating.writer.buffered();
    const padding = (8 - (json_bytes.len % 8)) % 8; // this ensures that the json is ends on an aligned boundary for the binary data to start
    // std.debug.print("padding {}\n", .{padding});
    try writer.writeInt(u64, @intCast(json_bytes.len + padding), .little);
    try writer.writeAll(json_bytes);
    try writer.splatByteAll(' ', padding);
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
};

pub const TensorMetaData = struct {
    name: []const u8,
    dtype: DType,
    shape: []const u64,
    relative_start: u64,
    relative_end: u64,
};

/// 8 bytes according to https://www.datacamp.com/blog/safetensors-format
pub const JsonHeader = u64; // TODO need to find a better name
