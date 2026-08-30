const std = @import("std");
const mem = std.mem;
const heap = std.heap;
const Io = std.Io;
const math = std.math;
const json = std.json;
const ascii = std.ascii;
const process = std.process;
const log = std.log;

const quantization = @import("quantization.zig");
const Weights = @import("Weights.zig");

pub const safetensors = @This();

pub const maximum_header_size: usize = 100 * 1024 * 1024;

pub fn parseSafetensorsStreaming(allocator: mem.Allocator, reader: *Io.Reader) !safetensors.Result {
    var result: Result = .init(allocator);
    errdefer result.deinit();
    const arena = result.arena.allocator();

    const header_size = try getHeaderSize(reader);
    result.tensors_start_seek_position = @sizeOf(u64) + header_size;

    var limited_json_buffer: [4096]u8 = undefined;
    var limited_reader: Io.Reader.Limited = .init(
        reader,
        .limited(@intCast(header_size)),
        &limited_json_buffer,
    );

    var json_reader: json.Reader = .init(
        arena,
        &limited_reader.interface,
    );

    if (try json_reader.next() != .object_begin) {
        return error.InvalidFileFormat;
    }

    const parse_options: json.ParseOptions = .{
        .max_value_len = @intCast(header_size),
        .allocate = .alloc_always,
    };

    var sequence: usize = 0;
    while (true) : (sequence += 1) {
        const object = try json_reader.nextAlloc(arena, .alloc_always);

        const name = switch (object) {
            .allocated_string => |value| value,
            .object_end => break,
            else => return error.InvalidFileFormat,
        };

        if (mem.eql(u8, "__metadata__", name)) {
            try json_reader.skipValue();
            continue;
        }

        const parsed = try json.innerParse(safetensors.Raw, arena, &json_reader, parse_options);

        if (parsed.data_offsets[0] > parsed.data_offsets[1]) {
            return error.InvalidFileFormat;
        }

        try result.values.append(arena, .{
            .sequence = sequence,
            .name = try arena.dupe(u8, name),
            .dtype = parsed.dtype,
            .shape = parsed.shape,
            .relative_start = parsed.data_offsets[0],
            .relative_end = parsed.data_offsets[1],
        });
    }

    return result;
}

fn getHeaderSize(reader: *Io.Reader) !u64 {
    const header_size = try reader.takeInt(u64, .little);

    if (header_size > maximum_header_size) {
        return error.InvalidHeaderSize;
    }

    return header_size;
}

fn stringifyHeader(tensor_metadata: std.MultiArrayList(MetaData).Slice, writer: *Io.Writer) Io.Writer.Error!void {
    var stringify: json.Stringify = .{
        .writer = writer,
        .options = .{},
    };

    const names = tensor_metadata.items(.name);
    const dtypes = tensor_metadata.items(.dtype);
    const shapes = tensor_metadata.items(.shape);
    const starts = tensor_metadata.items(.relative_start);
    const ends = tensor_metadata.items(.relative_end);

    try stringify.beginObject();

    for (names, dtypes, shapes, starts, ends) |name, dtype, shape, start, end| {
        try stringify.objectField(name);
        try stringify.write(Raw{
            .dtype = dtype,
            .shape = shape,
            .data_offsets = .{ start, end },
        });
    }

    try stringify.endObject();
}

pub fn writeHeader(allocator: mem.Allocator, tensor_metadata: std.MultiArrayList(MetaData).Slice, writer: *Io.Writer) !void {
    var allocating: Io.Writer.Allocating = .init(allocator);
    defer allocating.deinit();

    try stringifyHeader(tensor_metadata, &allocating.writer);
    const json_bytes = allocating.writer.buffered();
    const padding = (8 - (json_bytes.len % 8)) % 8;
    const header_size = json_bytes.len + padding;

    if (header_size > maximum_header_size) {
        return error.InvalidHeaderSize;
    }

    try writer.writeInt(u64, @intCast(header_size), .little);
    try writer.writeAll(json_bytes);
    try writer.splatByteAll(' ', padding);
}

const weights_suffixes = ".weight_packed";
const output_weight_suffix = ".weight";

pub fn buildMetadata(allocator: mem.Allocator, input: std.MultiArrayList(safetensors.MetaData).Slice, steps: []const Weights.Step) !safetensors.Result {
    std.debug.assert(input.len == steps.len);

    var output: safetensors.Result = .init(allocator);
    errdefer output.deinit();
    const arena = output.arena.allocator();

    const names = input.items(.name);
    const dtypes = input.items(.dtype);
    const shapes = input.items(.shape);
    const starts = input.items(.relative_start);
    const ends = input.items(.relative_end);

    var output_offset: u64 = 0;

    for (steps, names, dtypes, shapes, starts, ends) |step, input_name, input_dtype, input_shape, input_start, input_end| {
        std.debug.assert(input_end >= input_start);
        const input_byte_size = input_end - input_start;

        switch (step) {
            .cache_local_scale, .cache_global_scale => {
                continue;
            },
            .copy => {
                const output_end = try std.math.add(u64, output_offset, input_byte_size);

                try output.values.append(arena, .{
                    .sequence = output.values.len,
                    .name = try arena.dupe(u8, input_name),
                    .dtype = input_dtype,
                    .shape = try arena.dupe(u64, input_shape),
                    .relative_start = output_offset,
                    .relative_end = output_end,
                });
                output_offset = output_end;
            },
            .dequantize => {
                const basename = mem.cutSuffix(
                    u8,
                    input_name,
                    weights_suffixes,
                ) orelse return error.InvalidPackedWeightName;

                if (input_shape.len == 0) {
                    return error.InvalidPackedWeightShape;
                }

                const output_name = try mem.concat(arena, u8, &.{ basename, output_weight_suffix });
                const output_shape = try arena.dupe(u64, input_shape);
                output_shape[output_shape.len - 1] = try std.math.mul(u64, output_shape[output_shape.len - 1], 2);
                const output_byte_len = try std.math.mul(u64, input_byte_size, 8);
                const output_end = try std.math.add(u64, output_offset, output_byte_len);

                try output.values.append(arena, .{
                    .sequence = output.values.len,
                    .name = output_name,
                    .dtype = .F32,
                    .shape = output_shape,
                    .relative_start = output_offset,
                    .relative_end = output_end,
                });

                output_offset = output_end;
            },
        }
    }

    return output;
}

pub const Result = struct {
    arena: heap.ArenaAllocator,
    values: std.MultiArrayList(safetensors.MetaData),
    tensors_start_seek_position: usize = 0,

    var locked: bool = false;

    const SortCtx = struct {
        relative_starts: []const u64,

        pub fn lessThan(ctx: @This(), lhs_index: usize, rhs_index: usize) bool {
            return ctx.relative_starts[lhs_index] < ctx.relative_starts[rhs_index];
        }
    };

    pub fn init(allocator: mem.Allocator) Result {
        return .{
            .arena = .init(allocator),
            .values = .empty,
            .tensors_start_seek_position = 0,
        };
    }

    pub fn sortByRelativeStart(self: *Result) void {
        const values = self.values.slice();
        self.values.sort(SortCtx{
            .relative_starts = values.items(.relative_start),
        });
    }

    pub fn deinit(self: *Result) void {
        defer self.* = undefined;
        self.arena.deinit();
    }
};

const Raw = struct {
    dtype: quantization.DType,
    shape: []const u64,
    data_offsets: [2]u64,
};

pub const MetaData = struct {
    sequence: usize,
    name: []const u8,
    dtype: quantization.DType,
    shape: []const u64,
    relative_start: u64,
    relative_end: u64,
};
