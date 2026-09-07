const std = @import("std");
pub const packed_count: usize = 16;
pub const PackedWeights = [packed_count / 2]u8;
pub const DecodedWeights = [packed_count]f32;
pub const packed_size = @sizeOf(PackedWeights);
pub const decoded_size = @sizeOf(DecodedWeights);
const ByteVec = @Vector(packed_count, i8);

extern fn @"llvm.x86.ssse3.pshuf.b.128"(table: ByteVec, indices: ByteVec) ByteVec;
extern fn @"llvm.aarch64.neon.tbl1.v16i8"(table: ByteVec, indices: ByteVec) ByteVec;

pub const e2m1_lust: [16]i8 = .{
    0, 1,  2,  3,  4,  6,  8,  12,
    0, -1, -2, -3, -4, -6, -8, -12,
};

pub const e4m3_lut: [256]f32 = lut: {
    @setEvalBranchQuota(100_000);
    var values: [256]f32 = undefined;

    for (&values, 0..) |*slot, index| {
        const code: u8 = @intCast(index);

        if (code == 0 or code == 0x7f) {
            slot.* = 0;
            continue;
        }

        const exponent: u8 = (code >> 3) & 0x0f;
        const mantissa: u8 = (code & 0x07);
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
