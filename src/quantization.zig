const std = @import("std");
const builtin = @import("builtin");
const mem = std.mem;
const Io = std.Io;
const math = std.math;
const safetensors = @import("safetensors.zig");

pub const Dequantizer = struct {
    allocator: mem.Allocator,
    io: Io,
    input: *Io.Reader,
    output: *Io.Writer,
    ctx: *const safetensors.ParsedHeader,

    pub fn init(
        allocator: mem.Allocator,
        io: Io,
        input: *Io.Reader,
        output: *Io.Writer,
        ctx: *const safetensors.ParsedHeader,
    ) Dequantizer {
        return .{
            .allocator = allocator,
            .io = io,
            .input = input,
            .output = output,
            .ctx = ctx,
        };
    }

    pub fn dequantize(self: *Dequantizer, from: Scheme, to: Scheme) !void {
        if (from != .nvfp4 or to != .f32) {
            return error.UnsupportedConversion;
        }

        const tensors = self.ctx.header.tensors.slice();
        const operations = self.ctx.header.tensor_operations.items;
        std.debug.assert(operations.len == tensors.len);

        var pool: Pool = .{};
        pool.available = .init(&pool.available_buf);
        pool.decode = .init(&pool.decode_buf);
        pool.done = .init(&pool.done_buf);
        pool.chunks = try self.allocator.alloc(Chunk, Pool.count);

        errdefer {
            pool.close(self.io);
            self.allocator.free(pool.chunks);
        }

        for (pool.chunks) |*chunk| {
            try pool.available.putOne(self.io, chunk);
        }

        defer {
            pool.close(self.io);
            self.allocator.free(pool.chunks);
        }

        var writer_future = try self.io.concurrent(
            writeWorker,
            .{ self.io, self.output, &pool },
        );

        var workers: Io.Group = .init;
        var writer_done = false;
        var workers_done = false;

        defer {
            if (!workers_done or !writer_done) {
                pool.close(self.io);
                if (!workers_done) {
                    workers.cancel(self.io);
                }
                if (!writer_done) {
                    _ = writer_future.cancel(self.io) catch {};
                }
            }
        }

        for (0..worker_count) |_| {
            try workers.concurrent(self.io, decodeWorker, .{ self.io, &pool });
        }

        const cache = try self.allocator.alloc(WeightCache, tensors.len);
        defer {
            for (cache) |entry| {
                if (entry.local_scales) |scales| {
                    self.allocator.free(scales);
                }
            }
            self.allocator.free(cache);
        }
        @memset(cache, .{});

        var cursor: u64 = 0;
        var sequence: usize = 0;

        for (
            operations,
            tensors.items(.name),
            tensors.items(.info),
            0..,
        ) |operation, name, info, tensor_index| {
            const start = info.data_offsets[0];
            const size = info.data_offsets[1] - start;

            if (start > cursor) {
                try self.input.discardAll64(start - cursor);
            }

            switch (operation) {
                .copy => {
                    var remaining = size;
                    while (remaining > 0) {
                        const len: usize = @intCast(@min(
                            remaining,
                            Chunk.output_capacity,
                        ));
                        const chunk = try pool.takeAvailable(self.io);
                        chunk.sequence = sequence;
                        chunk.output_len = len;

                        try self.input.readSliceAll(
                            chunk.decoded_weights[0..len],
                        );
                        pool.done.putOne(self.io, chunk) catch |err| {
                            return pool.queueError(self.io, err);
                        };

                        sequence += 1;
                        remaining -= len;
                    }
                },
                .cache_local => {
                    const scales = try self.allocator.alloc(
                        u8,
                        @intCast(size),
                    );
                    errdefer self.allocator.free(scales);
                    try self.input.readSliceAll(scales);
                    cache[tensor_index].local_scales = scales;
                },
                .cache_global => {
                    if (size != @sizeOf(f32)) {
                        return error.InvalidGlobalScale;
                    }

                    const scale: f32 = @bitCast(
                        try self.input.takeInt(u32, .little),
                    );
                    if (!math.isFinite(scale) or scale <= 0) {
                        return error.InvalidGlobalScale;
                    }
                    cache[tensor_index].inverse_global_scale = 1.0 / scale;
                },
                .dequantize => {
                    const basename = mem.cutSuffix(
                        u8,
                        name,
                        safetensors.packed_suffix,
                    ) orelse return error.InvalidPackedWeightName;
                    const unit = self.ctx.header.tensor_units.get(basename) orelse
                        return error.MissingTensorUnit;
                    const local_index = unit.index_of_local_scale;
                    const scales = cache[local_index].local_scales orelse
                        return error.MissingCachedScale;
                    const inverse_scale = cache[unit.index_of_global_scale]
                        .inverse_global_scale orelse
                        return error.MissingCachedScale;

                    var remaining = size;
                    var scale_offset: usize = 0;

                    while (remaining > 0) {
                        const len: usize = @intCast(@min(remaining, Chunk.input_capacity));
                        std.debug.assert(len % @sizeOf(Nvfp4.PackedWeights) == 0);

                        const scale_count = len / @sizeOf(Nvfp4.PackedWeights);
                        const chunk = try pool.takeAvailable(self.io);

                        chunk.sequence = sequence;
                        chunk.input_len = len;
                        chunk.output_len = len * 8;
                        chunk.inverse_global_scale = inverse_scale;

                        try self.input.readSliceAll(chunk.packed_weights[0..len]);
                        @memcpy(chunk.local_scales[0..scale_count], scales[scale_offset..][0..scale_count]);

                        pool.decode.putOne(self.io, chunk) catch |err| {
                            return pool.queueError(self.io, err);
                        };

                        sequence += 1;
                        scale_offset += scale_count;
                        remaining -= len;
                    }

                    std.debug.assert(scale_offset == scales.len);
                    self.allocator.free(scales);
                    cache[local_index].local_scales = null;
                },
                .quantize => {
                    return error.UnsupportedConversion;
                },
            }

            cursor = info.data_offsets[1];
        }

        pool.decode.close(self.io);

        workers.await(self.io) catch |err| {
            return pool.getError(self.io) orelse err;
        };
        workers_done = true;

        pool.done.close(self.io);
        writer_future.await(self.io) catch |err| {
            return pool.getError(self.io) orelse err;
        };
        writer_done = true;

        if (pool.getError(self.io)) |err| {
            return err;
        }
    }
};

pub const Nvfp4 = struct {
    pub const packed_count: usize = 16;
    pub const PackedWeights = [packed_count / 2]u8;
    pub const DecodedWeights = [packed_count]f32;

    const ByteVec = @Vector(packed_count, i8);
    const BitsVec = @Vector(packed_count, u32);

    extern fn @"llvm.x86.ssse3.pshuf.b.128"(table: ByteVec, indices: ByteVec) ByteVec;
    extern fn @"llvm.aarch64.neon.tbl1.v16i8"(table: ByteVec, indices: ByteVec) ByteVec;

    pub const e2m1_lut: [16]i8 = .{
        0, 1,  2,  3,  4,  6,  8,  12,
        0, -1, -2, -3, -4, -6, -8, -12,
    };

    /// UE4M3 * 0.5 so the LUT matches kvalues_mxfp4 (2 * E2M1).
    pub const e4m3_lut: [256]f32 = lut: {
        @setEvalBranchQuota(100000);
        var values: [256]f32 = undefined;

        for (&values, 0..) |*slot, index| {
            const code: u8 = @intCast(index);

            if (code == 0 or code == 0x7f) {
                slot.* = 0;
                continue;
            }

            const exponent: u8 = (code >> 3) & 0x0f;
            const mantissa: u8 = code & 0x07;
            var value: f32 = undefined;

            if (exponent == 0) {
                value = @as(f32, @floatFromInt(mantissa)) / 1024.0;
            } else {
                value = 1.0 + @as(f32, @floatFromInt(mantissa)) / 8.0;
            }

            if (exponent != 0) {
                const power: i8 = @as(i8, @intCast(exponent)) - 8;

                if (power >= 0) {
                    for (0..@intCast(power)) |_| {
                        value *= 2.0;
                    }
                } else {
                    for (0..@intCast(-power)) |_| {
                        value *= 0.5;
                    }
                }
            }

            slot.* = value;
        }

        break :lut values;
    };

    pub fn decodePackedWeights(packed_weights: PackedWeights, scale: u8, inverse_global_scale: f32) DecodedWeights {
        const PackedVector = @Vector(8, u8);
        const CodeVector = @Vector(16, u8);
        const SignedVector = @Vector(16, i8);
        const FloatVector = @Vector(16, f32);
        const ShuffleMask = @Vector(16, i32);
        const nibble_mask: PackedVector = @splat(0x0f);
        const nibble_shift: PackedVector = @splat(4);

        const interleave_mask: ShuffleMask = .{
            0, -1, 1, -2, 2, -3, 3, -4,
            4, -5, 5, -6, 6, -7, 7, -8,
        };

        const quantized_weights: PackedVector = @bitCast(packed_weights);
        const low = quantized_weights & nibble_mask;
        const high = quantized_weights >> nibble_shift;
        const code: CodeVector = @shuffle(u8, low, high, interleave_mask);
        const code_values: [packed_count]u8 = @bitCast(code);
        const table: ByteVec = @bitCast(e2m1_lut);
        const indices: ByteVec = @bitCast(code);

        const e2m1: SignedVector = switch (builtin.cpu.arch) {
            .x86_64 => blk: {
                if (std.Target.x86.featureSetHas(builtin.cpu.features, .ssse3)) {
                    break :blk @"llvm.x86.ssse3.pshuf.b.128"(
                        table,
                        indices,
                    );
                }

                var values: [packed_count]i8 = undefined;

                for (code_values, 0..) |value, index| {
                    values[index] = e2m1_lut[value];
                }

                break :blk @bitCast(values);
            },
            .aarch64 => @"llvm.aarch64.neon.tbl1.v16i8"(table, indices),
            else => blk: {
                var values: [packed_count]i8 = undefined;

                for (code_values, 0..) |value, index| {
                    values[index] = e2m1_lut[value];
                }

                break :blk @bitCast(values);
            },
        };

        const converted_values: FloatVector = @floatFromInt(e2m1);
        const unsigned_values: BitsVec = @bitCast(converted_values);
        const negative_zero_code: CodeVector = @splat(0b1000);
        const negative_zero_bits: BitsVec = @splat(0x8000_0000);
        const values: FloatVector = @bitCast(@select(u32, code == negative_zero_code, negative_zero_bits, unsigned_values));
        const scaling_factor = e4m3_lut[scale] * inverse_global_scale;
        const scaling_vector: FloatVector = @splat(scaling_factor);
        const decoded = values * scaling_vector;

        return @bitCast(decoded);
    }
};

pub const Scheme = enum {
    nvfp4,
    f32,
};

const worker_count: usize = 4;
const chunks_per_worker: usize = 2;

const Chunk = struct {
    const input_capacity: usize = 64 * 1024;
    const scale_capacity: usize = input_capacity / 8;
    const output_capacity: usize = input_capacity * 8;

    packed_weights: [input_capacity]u8 = undefined,
    local_scales: [scale_capacity]u8 = undefined,
    decoded_weights: [output_capacity]u8 align(64) = undefined,
    sequence: usize = 0,
    input_len: usize = 0,
    output_len: usize = 0,
    inverse_global_scale: f32 = 0,
};

const Pool = struct {
    const count: usize = worker_count * chunks_per_worker;

    chunks: []Chunk = &.{},
    available_buf: [count]*Chunk = undefined,
    decode_buf: [count]*Chunk = undefined,
    done_buf: [count]*Chunk = undefined,
    available: Io.Queue(*Chunk) = undefined,
    decode: Io.Queue(*Chunk) = undefined,
    done: Io.Queue(*Chunk) = undefined,
    error_mutex: Io.Mutex = .init,
    first_error: ?anyerror = null,

    fn close(self: *Pool, io: Io) void {
        self.available.close(io);
        self.decode.close(io);
        self.done.close(io);
    }

    fn fail(self: *Pool, io: Io, err: anyerror) void {
        self.error_mutex.lockUncancelable(io);

        if (self.first_error == null) {
            self.first_error = err;
        }

        self.error_mutex.unlock(io);
        self.close(io);
    }

    fn getError(self: *Pool, io: Io) ?anyerror {
        self.error_mutex.lockUncancelable(io);
        defer self.error_mutex.unlock(io);
        return self.first_error;
    }

    inline fn takeAvailable(self: *Pool, io: Io) !*Chunk {
        return self.available.getOne(io) catch |err| {
            return self.queueError(io, err);
        };
    }

    fn queueError(self: *Pool, io: Io, err: anyerror) anyerror {
        if (err == error.Closed) {
            return self.getError(io) orelse error.Closed;
        } else {
            return err;
        }
    }
};

const WeightCache = struct {
    local_scales: ?[]u8 = null,
    inverse_global_scale: ?f32 = null,
};

fn closedOrCancel(err: anyerror, io: Io) Io.Cancelable!void {
    return switch (err) {
        error.Canceled => Io.recancel(io),
        error.Closed => {},
        else => unreachable,
    };
}

fn decodeWorker(io: Io, pool: *Pool) Io.Cancelable!void {
    while (true) {
        const chunk = pool.decode.getOne(io) catch |err| {
            return closedOrCancel(err, io);
        };

        const packed_size = @sizeOf(Nvfp4.PackedWeights);
        const decoded_size = @sizeOf(Nvfp4.DecodedWeights);
        const packed_bytes = chunk.packed_weights[0..chunk.input_len];
        const local_scales = chunk.local_scales[0 .. chunk.input_len / packed_size];
        const output = chunk.decoded_weights[0..chunk.output_len];

        std.debug.assert(packed_bytes.len % packed_size == 0);
        std.debug.assert(local_scales.len == packed_bytes.len / packed_size);
        std.debug.assert(output.len == local_scales.len * decoded_size);

        for (local_scales, 0..) |local_scale, index| {
            const decoded = Nvfp4.decodePackedWeights(
                packed_bytes[index * packed_size ..][0..packed_size].*,
                local_scale,
                chunk.inverse_global_scale,
            );

            const destination = output[index * decoded_size ..][0..decoded_size];

            if (comptime builtin.cpu.arch.endian() == .little) {
                @memcpy(destination, mem.asBytes(&decoded));
            } else {
                for (decoded, 0..) |value, value_index| {
                    mem.writeInt(
                        u32,
                        destination[value_index * 4 ..][0..4],
                        @bitCast(value),
                        .little,
                    );
                }
            }
        }

        pool.done.putOne(io, chunk) catch |err| {
            return closedOrCancel(err, io);
        };
    }
}

fn writeWorker(io: Io, writer: *Io.Writer, pool: *Pool) !void {
    var next: usize = 0;
    var pending: [Pool.count]?*Chunk = @splat(null);

    while (true) {
        const chunk = pool.done.getOne(io) catch |err| {
            switch (err) {
                error.Closed => break,
                error.Canceled => return Io.recancel(io),
            }
        };

        const slot = chunk.sequence % Pool.count;
        std.debug.assert(pending[slot] == null);
        pending[slot] = chunk;

        while (pending[next % Pool.count]) |ready| {
            std.debug.assert(ready.sequence == next);

            writer.writeAll(ready.decoded_weights[0..ready.output_len]) catch |err| {
                pool.fail(io, err);
                return err;
            };

            pending[next % Pool.count] = null;
            next += 1;

            pool.available.putOne(io, ready) catch |err| {
                switch (err) {
                    error.Closed => break,
                    error.Canceled => return Io.recancel(io),
                }
            };
        }
    }

    for (pending) |slot| {
        if (slot != null) {
            return pool.getError(io) orelse error.IncompleteOutput;
        }
    }

    writer.flush() catch |err| {
        pool.fail(io, err);
        return err;
    };
}

test "NVFP4 SIMD decode matches every E2M1 code" {
    const packed_weights: Nvfp4.PackedWeights = .{
        0x10, 0x32, 0x54, 0x76,
        0x98, 0xba, 0xdc, 0xfe,
    };
    const decoded = Nvfp4.decodePackedWeights(packed_weights, 0x40, 1.0);
    for (decoded, Nvfp4.e2m1_lut) |actual, expected_integer| {
        try std.testing.expectEqual(@as(f32, @floatFromInt(expected_integer)), actual);
    }
    try std.testing.expectEqual(@as(u32, 0x8000_0000), @as(u32, @bitCast(decoded[8])));
}
