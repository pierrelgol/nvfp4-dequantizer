const std = @import("std");
const Io = std.Io;
const Tensor = @import("Tensor.zig");
const json = std.json;
const mem = std.mem;
const heap = std.heap;
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
    tensor_units: std.StringArrayHashMapUnmanaged(Tensor.Unit) = .empty,
    tensor_operations: std.ArrayListUnmanaged(Tensor.Operation) = .empty,
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
                return ctx.slice.items(.info)[lhs].data_offsets[0] <
                    ctx.slice.items(.info)[rhs].data_offsets[0];
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

    fn indexTensorsByUnit(header: *safetensors.Header, allocator: mem.Allocator) !void {
        var scratch_buffer: [4096]u8 = undefined;
        var fba: heap.FixedBufferAllocator = .init(&scratch_buffer);

        for (header.tensors.items(.name), 0..) |name, index| {
            defer fba.reset();
            const basename = mem.cutSuffix(u8, name, packed_suffix) orelse continue;

            const local_scale_key = try mem.concat(
                fba.allocator(),
                u8,
                &.{ basename, local_scale_suffix },
            );

            const global_scale_key = try mem.concat(
                fba.allocator(),
                u8,
                &.{ basename, global_scale_suffix },
            );

            const local_scale_index = header.tensor_index.get(local_scale_key) orelse return error.MissingLocalScale;
            const global_scale_index = header.tensor_index.get(global_scale_key) orelse return error.MissingGrlobalScale;

            const gop = try header.tensor_units.getOrPut(allocator, basename);

            if (gop.found_existing) {
                return error.DuplicateTensorName;
            } else {
                gop.value_ptr.* = .{
                    .index_of_weights = index,
                    .index_of_local_scale = local_scale_index,
                    .index_of_global_scale = global_scale_index,
                };
            }
        }
    }

    fn indexTensorsByOperation(header: *safetensors.Header, allocator: mem.Allocator) !void {
        const operations = try allocator.alloc(Tensor.Operation, header.tensors.len);
        @memset(operations, .copy);

        var it = header.tensor_units.iterator();
        while (it.next()) |entry| {
            const unit = entry.value_ptr.*;
            operations[unit.index_of_weights] = .dequantize;
            operations[unit.index_of_local_scale] = .cache_local;
            operations[unit.index_of_global_scale] = .cache_global;
        }

        header.tensor_operations = .initBuffer(operations);
        header.tensor_operations.items = operations;
    }
};

pub const ParsedHeader = struct {
    arena: heap.ArenaAllocator,
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
            const parsed = try json.innerParse(?Tensor.Metadata, arena, &json_reader, options);
            result.header.metadata = parsed;
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
    try result.header.indexTensorsByUnit(arena);
    try result.header.indexTensorsByOperation(arena);

    return result;
}

pub fn buildDequantizedHeader(
    allocator: mem.Allocator,
    input: *const Header,
) !ParsedHeader {
    var result: ParsedHeader = .init(allocator, 0);
    errdefer result.deinit();
    const arena = result.arena.allocator();
    const tensors = input.tensors.slice();

    var output_offset: u64 = 0;
    for (input.tensor_operations.items, 0..) |operation, index| {
        const tensor = tensors.get(index);
        const input_size = tensor.info.data_offsets[1] -
            tensor.info.data_offsets[0];

        switch (operation) {
            .cache_local, .cache_global => {},
            .copy => {
                const output_end = try std.math.add(
                    u64,
                    output_offset,
                    input_size,
                );
                try result.header.tensors.append(arena, .{
                    .sequence = result.header.tensors.len,
                    .name = try arena.dupe(u8, tensor.name),
                    .info = .{
                        .dtype = tensor.info.dtype,
                        .shape = try arena.dupe(u64, tensor.info.shape),
                        .data_offsets = .{ output_offset, output_end },
                    },
                });
                output_offset = output_end;
            },
            .dequantize => {
                const basename = mem.cutSuffix(
                    u8,
                    tensor.name,
                    packed_suffix,
                ) orelse return error.InvalidPackedWeightName;
                if (tensor.info.shape.len == 0) {
                    return error.InvalidPackedWeightShape;
                }

                const output_shape = try arena.dupe(u64, tensor.info.shape);
                output_shape[output_shape.len - 1] = try std.math.mul(
                    u64,
                    output_shape[output_shape.len - 1],
                    2,
                );
                const output_size = try std.math.mul(u64, input_size, 8);
                const output_end = try std.math.add(
                    u64,
                    output_offset,
                    output_size,
                );
                try result.header.tensors.append(arena, .{
                    .sequence = result.header.tensors.len,
                    .name = try mem.concat(
                        arena,
                        u8,
                        &.{ basename, weight_suffix },
                    ),
                    .info = .{
                        .dtype = .F32,
                        .shape = output_shape,
                        .data_offsets = .{ output_offset, output_end },
                    },
                });
                output_offset = output_end;
            },
            .quantize => return error.UnsupportedConversion,
        }
    }

    return result;
}

pub fn writeHeader(allocator: mem.Allocator, header: *const Header, writer: *Io.Writer) !u64 {
    var allocating: Io.Writer.Allocating = .init(allocator);
    defer allocating.deinit();

    var stringify: json.Stringify = .{
        .writer = &allocating.writer,
        .options = .{},
    };

    try stringify.beginObject();
    const tensors = header.tensors.slice();

    for (0..tensors.len) |index| {
        const tensor = tensors.get(index);
        try stringify.objectField(tensor.name);
        try stringify.write(tensor.info);
    }

    try stringify.endObject();

    const json_bytes = allocating.writer.buffered();
    const padding = (8 - (json_bytes.len % 8)) % 8;
    const header_size = json_bytes.len + padding;

    if (header_size > Header.maximum_header_size) {
        return error.InvalidHeaderSize;
    }

    try writer.writeInt(u64, @intCast(header_size), .little);
    try writer.writeAll(json_bytes);
    try writer.splatByteAll(' ', padding);

    return @sizeOf(u64) + header_size;
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
