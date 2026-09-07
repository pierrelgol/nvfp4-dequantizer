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
    sequence: usize = 0,

    parsed: ?safetensors.ParsedHeader = null,
    steps: []Step = &.{},
    cache: []WeightCache = &.{},
    output_header: []u8 = &.{},
    output_header_idx: usize = 0,

    pool: ?*Pool = null,
    workers: Io.Group = .init,
    workers_started: bool = false,

    step_offset: usize = 0,
    step_index: usize = 0,
    current_idx: u64 = 0,
    next_write: usize = 0,
    in_flight: usize = 0,
    pending: [Pool.total_block_count]?*Block = @splat(null),
    emitting: ?*Block = null,
    emit_off: usize = 0,
    fill_eof: bool = false,
    payload_done: bool = false,
    decode_closed: bool = false,

    const vtable: Io.Reader.VTable = .{ .stream = stream };

    pub fn init(
        allocator: mem.Allocator,
        io: Io,
        input: *Io.Reader,
        buffer: []u8,
    ) Dequantizer {
        return .{
            .interface = .{
                .vtable = &vtable,
                .buffer = buffer,
                .seek = 0,
                .end = 0,
            },
            .input = input,
            .allocator = allocator,
            .io = io,
        };
    }

    pub fn deinit(self: *Dequantizer) void {
        defer self.* = undefined;

        self.stop();
        if (self.pool) |pool| {
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
        self.allocator.free(self.output_header);

        if (self.parsed) |*parsed| {
            parsed.deinit();
        }
    }

    pub fn reader(self: *Dequantizer) *Io.Reader {
        return &self.interface;
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

    pub fn stream(r: *Io.Reader, w: *Io.Writer, limit: Io.Limit) Io.Reader.StreamError!usize {
        const self: *Dequantizer = @fieldParentPtr("interface", r);

        if (self.maybe_err) |err| {
            return self.fail(err);
        }

        if (limit == .nothing) {
            return 0;
        }

        if (self.parsed == null) {
            try self.initializeStream();
        }

        var written: usize = 0;

        if (self.output_header_idx < self.output_header.len) {
            const n = try w.write(limit.sliceConst(self.output_header[self.output_header_idx..]));

            if (n == 0) {
                return error.WriteFailed;
            }

            self.output_header_idx += n;
            written += n;
            limit = limit.subtract(n) orelse return written;

            if (limit == .nothing or self.output_header_idx < self.output_header.len) {
                return written;
            }
        }

        if (self.payload_done) {
            if (written == 0) {
                return error.EndOfStream;
            } else {
                return written;
            }
        }

        while (limit != .nothing) {
            try self.queueAvailableBlocks();
            try self.collectCompletedBlock(false);

            if (self.emitting == null) {
                if (self.pending[self.next_write % Pool.total_block_count]) |block| {
                    if (block.sequence == self.next_write) {
                        self.emitting = block;
                        self.emit_off = 0;
                        self.pending[self.next_write % Pool.total_block_count] = null;
                    }
                }
            }

            if (self.emitting) |block| {
                const rest = block.output_bytesput_bytes[self.emit_off..block.output_bytesput_len];

                if (rest.len == 0) {
                    try self.recycleEmittedBlock();
                    continue;
                }

                const n = try w.write(limit.sliceConst(rest));

                if (n == 0) {
                    return error.WriteFailed;
                }

                self.emit_off += n;
                written += n;
                limit = limit.subtract(n) orelse return written;

                if (self.emit_off == block.output_bytesput_len) {
                    try self.recycleEmittedBlock();
                }

                if (limit == .nothing) {
                    return written;
                }

                continue;
            }

            if (self.fill_eof and self.in_flight == 0) {
                self.closeDecode();
                self.payload_done = true;
                break;
            }

            try self.collectCompletedBlock(true);
        }

        if (written == 0) {
            if (self.payload_done) {
                return error.EndOfStream;
            } else {
                return error.ReadFailed;
            }
        }
        return written;
    }

    fn recycleEmittedBlock(self: *Dequantizer) !void {
        const block = self.emitting.?;
        self.emitting = null;
        self.next_write += 1;
        self.in_flight -= 1;
        const pool = self.pool.?;

        pool.available.putOne(self.io, block) catch |err| {
            return pool.queueError(self.io, err);
        };
    }

    fn collectCompletedBlock(self: *Dequantizer, wait: bool) !void {
        const pool = self.pool.?;
        var slot: [1]*Block = undefined;

        const n = if (wait) blk: {
            slot[0] = pool.decompressed.getOne(self.io) catch |err| {
                return switch (err) {
                    error.Closed => pool.getError(self.io) orelse error.Closed,
                    error.Canceled => error.Canceled,
                };
            };

            break :blk 1;
        } else pool.decompressed.get(self.io, &slot, 0) catch |err| {
            return switch (err) {
                error.Closed => pool.getError(self.io) orelse error.Closed,
                error.Canceled => error.Canceled,
            };
        };

        if (n == 0) {
            return;
        }

        const block = slot[0];
        const index = block.sequence % Pool.total_block_count;

        std.debug.assert(self.pending[index] == null);
        self.pending[index] = block;
    }

    fn closeDecode(self: *Dequantizer) void {
        if (self.decode_closed) {
            return;
        }

        if (self.pool) |pool| {
            pool.decodable.close(self.io);
        }

        self.decode_closed = true;
    }

    fn initializeStream(self: *Dequantizer) !void {
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

        self.output_header = try serializeF32Header(self.allocator, parsed, self.steps);
        errdefer {
            self.allocator.free(self.output_header);
            self.output_header = &.{};
        }

        const pool = try self.allocator.create(Pool);
        errdefer self.allocator.destroy(pool);

        pool.* = .{};
        pool.available = .init(&pool.available_block_buffer);
        pool.decodable = .init(&pool.decodable_block_buffer);
        pool.decompressed = .init(&pool.decompressed_block_buffer);
        pool.blocks = try self.allocator.alloc(Block, Pool.total_block_count);
        errdefer self.allocator.free(pool.blocks);
        self.pool = pool;
        errdefer {
            self.stop();
            self.pool = null;
        }

        for (pool.blocks) |*block| {
            try pool.available.putOne(self.io, block);
        }

        const n_workers = workerCount();
        for (0..n_workers) |_| {
            try self.workers.concurrent(self.io, decodeWorker, .{ self.io, pool });
            self.workers_started = true;
        }
    }

    fn readAndParseHeader(self: *Dequantizer) !safetensors.ParsedHeader {
        return safetensors.parse(self.allocator, self.input);
    }

    fn queueAvailableBlocks(self: *Dequantizer) !void {
        if (self.fill_eof) {
            return;
        }

        const pool = self.pool.?;
        while (true) {
            var slot: [1]*Block = undefined;

            const n = pool.available.get(self.io, &slot, 0) catch |err| {
                return pool.queueError(self.io, err);
            };

            if (n == 0) {
                return;
            }

            switch (try self.fillNextBlock(slot[0])) {
                .eof => {
                    pool.available.putOne(self.io, slot[0]) catch |err| return pool.queueError(self.io, err);
                    self.fill_eof = true;
                    self.closeDecode();

                    return;
                },
                .filled => {
                    self.in_flight += 1;
                    const block = slot[0];

                    if (block.kind == .dequant) {
                        pool.decodable.putOne(self.io, block) catch |err| return pool.queueError(self.io, err);
                    } else {
                        pool.decompressed.putOne(self.io, block) catch |err| return pool.queueError(self.io, err);
                    }
                },
            }
        }
    }

    fn fillNextBlock(self: *Dequantizer, block: *Block) !enum { filled, eof } {
        while (self.step_index < self.steps.len) {
            const step = self.steps[self.step_index];

            if (self.current_idx < step.src_start) {
                self.input.discardAll64(step.src_start - self.current_idx) catch |err| {
                    if (err == error.EndOfStream) {
                        return error.TruncatedPayload;
                    } else {
                        return err;
                    }
                };

                self.current_idx = step.src_start;
            }

            if (self.step_offset == step.src_len) {
                self.step_index += 1;
                self.step_offset = 0;

                continue;
            }

            const remaining = step.src_len - self.step_offset;
            switch (step.kind) {
                .cache_local => {
                    if (self.cache[self.step_index].local_scales == null) {
                        self.cache[self.step_index].local_scales = try self.allocator.alloc(u8, @intCast(step.src_len));
                    }

                    const destination = self.cache[self.step_index].local_scales.?;
                    const offset: usize = @intCast(self.step_offset);

                    self.input.readSliceAll(destination[offset..][0..@intCast(remaining)]) catch |err| {
                        if (err == error.EndOfStream) {
                            return error.TruncatedPayload;
                        } else {
                            return err;
                        }
                    };

                    self.current_idx += remaining;
                    self.step_offset = step.src_len;

                    continue;
                },
                .cache_global => {
                    var buf: [4]u8 = undefined;

                    std.debug.assert(step.src_len == 4);
                    self.input.readSliceAll(buf[0..4]) catch |err| {
                        if (err == error.EndOfStream) {
                            return error.TruncatedPayload;
                        } else {
                            return err;
                        }
                    };

                    try storeGlobalScale(&self.cache[self.step_index], buf);

                    self.current_idx += 4;
                    self.step_offset = 4;

                    continue;
                },
                .copy => {
                    const n: usize = @intCast(@min(remaining, Block.output_capacity));

                    self.input.readSliceAll(block.output_bytesput_bytes[0..n]) catch |err| {
                        if (err == error.EndOfStream) {
                            return error.TruncatedPayload;
                        } else {
                            return err;
                        }
                    };

                    block.sequence = self.sequence;
                    block.output_bytesput_len = n;
                    block.packed_len = n;
                    block.kind = .copy;
                    self.sequence += 1;
                    self.current_idx += n;
                    self.step_offset += n;

                    return .filled;
                },
                .dequant => {
                    const n: usize = @intCast(@min(remaining, Block.packed_capacity));
                    std.debug.assert(n % nvfp4.packed_size == 0);

                    self.input.readSliceAll(block.packed_bytes[0..n]) catch |err| {
                        if (err == error.EndOfStream) {
                            return error.TruncatedPayload;
                        } else {
                            return err;
                        }
                    };

                    const scale_count = n / nvfp4.packed_size;
                    const scale_offset: usize = @intCast(self.step_offset / nvfp4.packed_size);
                    const scales = self.cache[step.scale_index].local_scales orelse return error.MissingCachedScale;
                    const inv = self.cache[step.global_index].inverse_global_scale orelse return error.MissingCachedScale;
                    @memcpy(block.scale_bytes[0..scale_count], scales[scale_offset..][0..scale_count]);

                    block.sequence = self.sequence;
                    block.packed_len = n;
                    block.output_bytesput_len = n * 8;
                    block.inverse_global_scale = inv;
                    block.kind = .dequant;
                    self.sequence += 1;
                    self.current_idx += n;
                    self.step_offset += n;

                    if (self.step_offset == step.src_len) {
                        self.allocator.free(scales);
                        self.cache[step.scale_index].local_scales = null;
                    }

                    return .filled;
                },
            }
        }
        return .eof;
    }
};

pub const Step = struct {
    kind: enum { copy, cache_local, cache_global, dequant },
    src_start: u64,
    src_len: u64,
    dst_len: u64,
    scale_index: u32 = 0,
    global_index: u32 = 0,
};

pub const WeightCache = struct {
    local_scales: ?[]u8 = null,
    inverse_global_scale: ?f32 = null,
};

fn storeGlobalScale(cache: *WeightCache, bytes: [4]u8) !void {
    const scale: f32 = @bitCast(mem.readInt(u32, &bytes, .little));

    if (!math.isFinite(scale) or scale <= 0) {
        return error.InvalidGlobalScale;
    }

    cache.inverse_global_scale = 1.0 / scale;
}

fn workerCount() usize {
    const n = std.Thread.getCpuCount() catch Pool.total_workers;
    return @max(1, @min(Pool.total_workers, n));
}

pub const Block = struct {
    const packed_capacity: usize = 64 * 1024;
    const scale_capacity: usize = packed_capacity / 8;
    const output_capacity: usize = packed_capacity * 8;

    packed_bytes: [packed_capacity]u8 = undefined,
    packed_len: usize = 0,

    scale_bytes: [scale_capacity]u8 = undefined,
    scale_len: usize = 0,

    output_bytes: [output_capacity]u8 align(64) = undefined,
    output_len: usize = 0,

    sequence: usize = 0,
    inv_scale: f32 = 0,
    kind: enum { copy, dequant } = .copy,

    pub const init: Block = .{};
};

pub const Pool = struct {
    pub const total_workers: usize = 4;
    pub const block_per_workers: usize = 2;
    pub const total_block_count: usize = total_workers * block_per_workers;

    blocks: []Block = &.{},
    available: Io.Queue(*Block) = undefined,
    available_block_buffer: [total_block_count]*Block = undefined,

    decodable: Io.Queue(*Block) = undefined,
    decodable_block_buffer: [total_block_count]*Block = undefined,

    decompressed: Io.Queue(*Block) = undefined,
    decompressed_block_buffer: [total_block_count]*Block = undefined,

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

fn decodeWorker(io: Io, pool: *Pool) Io.Cancelable!void {
    _ = io;
    _ = pool;
}

fn buildStepsList(allocator: mem.Allocator, header: *const safetensors.Header) ![]Step {
    _ = allocator;
    _ = header;
}

fn serializeF32Header(allocator: mem.Allocator, parsed: *const safetensors.ParsedHeader, steps: []const Step) ![]u8 {
    _ = allocator;
    _ = parsed;
    _ = steps;
}
