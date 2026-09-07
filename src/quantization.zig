const std = @import("std");
const mem = std.mem;
const Io = std.Io;
const math = std.math;
const json = std.json;
const heap = std.heap;
const safetensors = @import("safetensors.zig");
const nvfp4 = @import("nvfp4.zig");
const Tensor = @import("Tensor.zig");

pub const Dequantizer = struct {
    allocator: mem.Allocator,
    io: std.Io,
    input: *Io.Reader,
    interface: Io.Reader,
    maybe_err: ?anyerror = null,
    parsed: safetensors.ParsedHeader,
    steps: []Step = &.{},
    cache: []WeightCache = &.{},
    pool: ?*Pool = null,
    workers: Io.Group = .init,
    workers_started: bool = false,
    sequence: usize = 0,

    const vtable: Io.Reader.VTable = &.{
        .stream = stream,
    };

    pub fn init(allocator: mem.Allocator, io: std.Io, input: *Io.Reader, buffer: []u8) Dequantizer {
        return .{
            .allocator = allocator,
            .io = io,
            .input = input,
            .interface = .{
                .vtable = &vtable,
                .buffer = buffer,
                .seek = 0,
                .end = 0,
            },
            .sequence = 0,
        };
    }

    pub fn deinit(self: *Dequantizer) void {
        defer self.* = undefined;
        self.stop();

        if (self.pool) |pool| {
            defer self.pool = undefined;
            self.allocator.free(pool.blocks);
            self.allocator.destroy(pool);
        }

        for (self.cache) |entry| {
            if (entry.local_scales) |scales| {
                self.allocator.free(scales);
            }
        }

        self.allocator.free(self.cache);
        self.allocator.free(self.steps);

        if (self.parsed) |*parsed| {
            parsed.deinit();
        }
    }

    pub fn reader(self: *Dequantizer) *Io.Reader {
        return &self.interface;
    }

    pub fn stream(r: *Io.Reader, w: *Io.Writer, limit: Io.Limit) Io.Reader.StreamError!usize {
        const self: *Dequantizer = @fieldParentPtr("interface", r);

        if (self.maybe_err) |err| {
            return self.fail(err);
        }

        if (limit == .nothing) {
            return 0;
        }

        if (self.parsed == null) {
            try self.parse();
        }

        _ = w;
    }

    pub fn stop(self: *Dequantizer) void {
        if (self.pool) |pool| {
            pool.close(self.io);
        }

        if (self.workers_started) {
            self.workers.cancel(self.io);
            self.workers_started = false;
        }
    }

    pub fn fail(self: *Dequantizer, err: anyerror) Io.Reader.StreamError {
        if (self.maybe_err == null) {
            self.maybe_err = err;
        }

        if (self.pool) |pool| {
            pool.fail(self.io, err);
        }

        switch (err) {
            error.WriteFailed, error.EndOfStream => |e| return e,
            else => return error.ReadFailed,
        }
    }

    fn parse(self: *Dequantizer) !void {
        self.parsed = self.readAndParseHeader() catch |err| {
            return switch (err) {
                error.EndOfStream, error.UnexpectedEndOfInput => error.TruncatedHeader,
                else => err,
            };
        };

        const parsed = &self.parsed.?;
        self.steps = try buildStepsList(self.allocator, &parsed.header);
        errdefer {
            self.allocator.free(self.steps);
            self.steps = &.{};
        }

        self.cache = try self.allocator.alloc(WeightCache, parsed.header.tensors.len);
        errdefer {
            self.allocator.free(self.cache);
            self.cache = &.{};
        }
        @memset(self.cache, .{});

        self.header_out = try serializeF32Header(self.allocator, parsed, self.steps);
        errdefer {
            self.allocator.free(self.header_out);
            self.header_out = &.{};
        }

        const pool = try self.allocator.create(Pool);
        errdefer self.allocator.destroy(pool);

        pool.* = .{};
        pool.available = .init(&pool.available_buf);
        pool.decode = .init(&pool.decode_buf);
        pool.done = .init(&pool.done_buf);
        pool.tiles = try self.allocator.alloc(Block, Pool.total_block_count);

        errdefer self.allocator.free(pool.tiles);

        self.pool = pool;
        errdefer {
            self.stop();
            self.pool = null;
        }

        for (pool.tiles) |*tile| {
            try pool.available.putOne(self.io, tile);
        }

        for (0..Pool.total_worker) |_| {
            try self.workers.concurrent(self.io, decodeWorker, .{ self.io, pool });
            self.workers_started = true;
        }
    }

    fn buildStepsList(allocator: mem.Allocator, header: *safetensors.Header) !void {
        _ = allocator;
        _ = header;
    }

    fn serializeF32Header(allocator: mem.Allocator, parsed: safetensors.ParsedHeader, steps: []Step) void {
        _ = allocator;
        _ = parsed;
        _ = steps;
    }

    fn decodeWorker() void {}
};

pub const WeightCache = struct {
    local_scales: ?[]u8 = null,
    inverse_global_scale: f32 = null,
};

pub const Step = enum {};

pub const Block = struct {
    const packed_capacity: usize = 64 * 1024;
    const scale_capacity: usize = packed_capacity / 8;
    const output_capacity: usize = packed_capacity / 8;

    sequence: usize = 0,
    kind: enum { copy, dequant } = .copy,

    packed_bytes: [packed_capacity]u8 = undefined,
    packed_len: usize = 0,

    output_bytes: [output_capacity]u8 = undefined,
    output_len: usize = 0,

    scale_bytes: [scale_capacity]u8 = undefined,
    scale_len: usize = 0,

    pub const init: Block = .{};
};

pub const Pool = struct {
    pub const total_workers: usize = 4;
    pub const block_per_workers: usize = 4;
    pub const total_block_count: usize = total_workers * block_per_workers;

    blocks: []Block = &.{},

    available_block_buffer: [total_block_count]*Block = @splat(undefined),
    available: Io.Queue(*Block) = undefined,

    decodable_block_buffer: [total_block_count]*Block = @splat(undefined),
    decodable: Io.Queue(*Block) = undefined,

    decompressed_block_buffer: [total_block_count]*Block = @splat(undefined),
    decompressed: Io.Queue(*Block) = undefined,

    error_mutex: Io.Mutex = .init,
    maybe_error: ?anyerror = null,

    pub const init: Pool = .{};

    fn close(self: *Pool, io: std.Io) void {
        self.available.close(io);
        self.decodable.close(io);
        self.decompressed.close(io);
    }

    fn fail(self: *Pool, io: std.Io, err: anyerror) void {
        defer self.close(io);

        self.error_mutex.lockUncancelable(io);
        if (self.maybe_error == null) {
            self.maybe_error = err;
        }
        self.error_mutex.unlock(io);
    }

    fn queueError(self: *Pool, io: Io, err: anyerror) anyerror {
        if (err == error.Closed) {
            return self.getError(io) orelse error.Closed;
        } else {
            return err;
        }
    }

    fn getError(self: *Pool, io: std.Io) ?anyerror {
        self.error_mutex.lockUncancelable(io);
        defer self.error_mutex.unlock(io);

        return self.maybe_error;
    }
};
