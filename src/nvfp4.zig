const std = @import("std");
const builtin = @import("builtin");
const mem = std.mem;

/// NVFP4 block decode: 8 packed bytes (16 E2M1 codes) plus one UE4M3 scale.
pub const packed_count: usize = 16;
pub const PackedWeights = [packed_count / 2]u8;
pub const DecodedWeights = [packed_count]f32;
pub const packed_size: usize = @sizeOf(PackedWeights);
pub const decoded_size: usize = @sizeOf(DecodedWeights);

const ByteVec = @Vector(packed_count, i8);

extern fn @"llvm.x86.ssse3.pshuf.b.128"(table: ByteVec, indices: ByteVec) ByteVec;
extern fn @"llvm.aarch64.neon.tbl1.v16i8"(table: ByteVec, indices: ByteVec) ByteVec;

pub const e2m1_lut: [16]i8 = .{
    0, 1,  2,  3,  4,  6,  8,  12,
    0, -1, -2, -3, -4, -6, -8, -12,
};

// UE4M3 * 0.5 so the LUT matches kvalues_mxfp4 (2 * E2M1).
// Derived from llama.cpp/ggml UE4M3 conversion.
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
    const BitsVec = @Vector(16, u32);
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

fn storeBlock(destination: []u8, decoded: DecodedWeights) void {
    if (comptime builtin.cpu.arch.endian() == .little) {
        @memcpy(destination, mem.asBytes(&decoded));
    } else {
        for (decoded, 0..) |value, value_index| {
            mem.writeInt(u32, destination[value_index * 4 ..][0..4], @bitCast(value), .little);
        }
    }
}

pub fn decodeBlocks(packed_bytes: []const u8, scales: []const u8, inverse_global_scale: f32, out: []u8) void {
    std.debug.assert(packed_bytes.len == scales.len * packed_size);
    std.debug.assert(out.len == scales.len * decoded_size);

    var index: usize = 0;
    while (index + 4 <= scales.len) : (index += 4) {
        const packed_off = index * packed_size;
        const out_off = index * decoded_size;
        if (packed_off + 8 * packed_size <= packed_bytes.len) {
            @prefetch(packed_bytes.ptr + packed_off + 4 * packed_size, .{
                .rw = .read,
                .locality = 3,
                .cache = .data,
            });
        }

        inline for (0..4) |lane| {
            storeBlock(
                out[out_off + lane * decoded_size ..][0..decoded_size],
                decodePackedWeights(
                    packed_bytes[packed_off + lane * packed_size ..][0..packed_size].*,
                    scales[index + lane],
                    inverse_global_scale,
                ),
            );
        }
    }

    while (index < scales.len) : (index += 1) {
        storeBlock(
            out[index * decoded_size ..][0..decoded_size],
            decodePackedWeights(
                packed_bytes[index * packed_size ..][0..packed_size].*,
                scales[index],
                inverse_global_scale,
            ),
        );
    }
}

test "NVFP4 SIMD decode matches every E2M1 code" {
    const packed_weights: PackedWeights = .{
        0x10, 0x32, 0x54, 0x76,
        0x98, 0xba, 0xdc, 0xfe,
    };

    const decoded = decodePackedWeights(packed_weights, 0x40, 1.0);

    for (decoded, e2m1_lut) |actual, expected_integer| {
        try std.testing.expectEqual(@as(f32, @floatFromInt(expected_integer)), actual);
    }
}

test "NVFP4 applies local and global scales" {
    const packed_weights: PackedWeights = @splat(0x11);
    const decoded = decodePackedWeights(packed_weights, 0x48, 0.25);
    for (decoded) |value| {
        try std.testing.expectEqual(@as(f32, 0.5), value);
    }
}

test "E4M3 lookup has expected basic values" {
    try std.testing.expectEqual(@as(f32, 0), e4m3_lut[0]);
    try std.testing.expectEqual(@as(f32, 0), e4m3_lut[0x7f]);
    try std.testing.expectEqual(@as(f32, 1), e4m3_lut[0x40]);
    try std.testing.expectEqual(@as(f32, 2), e4m3_lut[0x48]);
}

test "decodeTiles matches decodePackedWeights" {
    var packed_bytes: [32]u8 = undefined;
    var scales: [4]u8 = undefined;
    for (&packed_bytes, 0..) |*slot, i| slot.* = @truncate(i * 17);
    for (&scales, 0..) |*slot, i| slot.* = if (i == 0) 0x40 else 0x48;

    var out: [256]u8 = undefined;
    decodeBlocks(&packed_bytes, &scales, 1.0, &out);

    for (scales, 0..) |scale, i| {
        const expected = decodePackedWeights(packed_bytes[i * 8 ..][0..8].*, scale, 1.0);
        const got: DecodedWeights = @bitCast(out[i * 64 ..][0..64].*);
        for (expected, got) |e, g| {
            try std.testing.expectEqual(e, g);
        }
    }
}
