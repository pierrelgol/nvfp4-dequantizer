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
};

pub const Header = struct {
    tensors: Tensor.Map = .{},
    metadata: ?Tensor.Metadata = null,

    pub const maximum_header_size = 100 * 1024 * 1024;

    pub fn format(
        self: @This(),
        writer: *std.Io.Writer,
    ) std.Io.Writer.Error!void {
        try json.Stringify.value(self, .{ .whitespace = .minified }, writer);
    }
};

pub const Result = struct {
    arena: heap.ArenaAllocator,
    header: safetensors.Header,
    header_size: u64 = 0,
    tensors_start_offset: u64 = 0,

    pub fn init(allocator: mem.Allocator, header_size: u64) Result {
        return .{
            .arena = .init(allocator),
            .header = .{},
            .header_size = header_size,
            .tensors_start_offset = header_size + @sizeOf(u64),
        };
    }

    pub fn deinit(self: *Result) void {
        defer self.* = undefined;
        self.arena.deinit();
    }
};

pub fn parse(allocator: mem.Allocator, reader: *Io.Reader) !safetensors.Result {
    const header_size = try reader.takeInt(u64, .little);

    if (header_size > Header.maximum_header_size) {
        return error.InvalidHeaderSize;
    }

    var result: Result = .init(allocator, header_size);
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

    while (true) {
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

        const pre = try json_reader.peekNextTokenType();
        const parsed = json.innerParse(Tensor.Info, arena, &json_reader, options) catch |err| {
            std.debug.print("{s}\n", .{@tagName(pre)});
            return err;
        };

        if (parsed.data_offsets[0] > parsed.data_offsets[1]) {
            return error.InvalidTensorOffsets;
        }

        const gop = try result.header.tensors.map.getOrPut(arena, object_name);

        if (gop.found_existing) {
            return error.DuplicateTensorName;
        } else {
            gop.value_ptr.* = parsed;
        }
    }

    return result;
}
