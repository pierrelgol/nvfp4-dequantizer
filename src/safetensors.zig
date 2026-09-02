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

pub const Header = struct {
    tensors: std.MultiArrayList(Tensor) = .empty,
    tensor_index: std.StringArrayHashMapUnmanaged(Tensor.Index) = .empty,
    tensor_units: std.StringArrayHashMapUnmanaged(Tensor.Unit) = .empty,
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

    return result;
}
