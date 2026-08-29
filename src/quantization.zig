const std = @import("std");
const mem = std.mem;
const heap = std.heap;
const Io = std.Io;
const math = std.math;
const json = std.json;
const ascii = std.ascii;
const process = std.process;

pub const Format = enum {
    none,
    nvfp4,
    f32,

    pub fn formatFromString(str: []const u8) ?Format {
        if (mem.eql(u8, "none", str)) {
            return null;
        } else if (mem.eql(u8, "nvfp4", str)) {
            return Format.nvfp4;
        } else if (mem.eql(u8, "f32", str)) {
            return Format.f32;
        } else {
            return null;
        }
    }
};

/// source that helped : https://github.com/pytorch/pytorch/blob/main/torch/headeronly/util/Float8_e4m3fn.h
/// also this is peak: https://github.com/ggml-org/ggml/blob/master/src/ggml-quants.c#L589
/// and of course this : https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
pub const Nvfp4 = struct {
    pub const total_packed_weights: usize = 16;
    pub const PackedWeights = [total_packed_weights >> 1]u8; // 8 * u4 = 16 u4 in E2m1
    pub const DecodedWeights = [total_packed_weights]f32;

    pub fn decodePackedWeights(packed_weights: PackedWeights, scale: u8, weight_global_scale: f32) DecodedWeights {
        var result: DecodedWeights = undefined;
        const scaling_factor: f32 = decodeE4M3(scale) / weight_global_scale;

        for (packed_weights, 0..) |w, i| {
            // assumes interleaved
            result[i * 2] = decodeE2M1(@as(u4, @truncate(w))) * scaling_factor; // first 4 bits
            result[i * 2 + 1] = decodeE2M1(@as(u4, @truncate(w >> 4))) * scaling_factor; // and the second ones;
        }

        return result;
    }

    /// used this as inspiration: https://github.com/ggml-org/ggml/blob/master/src/ggml-quants.c#L589
    fn decodeE2M1(w: u4) f32 {
        const magnitude = [_]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
        const value = magnitude[w & 0x7];
        return if (w & 0x8 == 0) value else -value;
    }

    fn decodeE4M3(s: u8) f32 {
        const sign: f32 = if (s & 0x80 == 0) 1 else -1;
        const exponent = (s >> 3) & 0x0f;
        const mantissa = s & 0x07;

        if (exponent == 0) {
            return sign * @as(f32, @floatFromInt(mantissa)) / 512.0;
        }
        const fraction = 1.0 + @as(f32, @floatFromInt(mantissa)) / 8.0;
        return sign * std.math.scalbn(fraction, @as(i32, exponent) - 7);
    }
};

pub const DType = enum {
    BOOL,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F4,
    F6_E2M3,
    F6_E3M2,
    F8_E4M3,
    F8_E5M2,
    F8_E8M0,
    F16,
    BF16,
    F32,
    F64,
};
