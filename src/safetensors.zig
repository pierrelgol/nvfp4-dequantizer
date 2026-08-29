const std = @import("std");
const mem = std.mem;
const heap = std.heap;
const Io = std.Io;
const math = std.math;
const json = std.json;
const ascii = std.ascii;
const process = std.process;
const quantization = @import("quantization.zig");
const log = std.log;
pub const safetensors = @This();

pub const maximum_header_size: usize = 100 * 1024 * 1024;

pub fn parseSafetensorsStreaming(allocator: mem.Allocator, reader: *Io.Reader) !safetensors.Result {
    var result: Result = .init(allocator);
    errdefer result.deinit();
    const arena = result.arena.allocator();

    const header_size = try getHeaderSize(reader);
    result.tensors_start_seek_position = @sizeOf(u64) + header_size;

    var limited_json_buffer: [std.heap.pageSize()]u8 = undefined;
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

pub const Result = struct {
    arena: heap.ArenaAllocator,
    values: std.MultiArrayList(safetensors.MetaData),
    tensors_start_seek_position: usize = 0,

    var locked: bool = false;

    /// those are to ensure poionter stability for the
    /// next step of the pipeline since MAL doesn't have
    /// lockPointers
    pub fn lock(_: *const Result) void {
        std.debug.assert(locked == false);
    }

    pub fn unlock(_: *const Result) void {
        std.debug.assert(locked);
    }

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
