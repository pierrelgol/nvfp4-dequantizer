const std = @import("std");
pub const Tensor = @This();

const Map = std.StringArrayHashMapUnmanaged;
pub const Metadata = std.StringArrayHashMapUnmanaged;

pub const Info = struct {
    dtype: Dtype,
    shape: Shape,
    data_offsets: DataOffsets,
};

pub const Shape = []u64;
pub const DataOffsets = [2]u64;

/// stolen from safetensors/lib/tensor.rs
pub const Dtype = enum {
    // Boolan type
    BOOL,
    // MXF4 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
    F4,
    // MXF6 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
    F6_E2M3,
    // MXF6 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
    F6_E3M2,
    // Unsigned byte
    U8,
    // Signed byte
    I8,
    // FP8 <https://arxiv.org/pdf/2209.05433.pdf>_
    F8_E5M2,
    // FP8 <https://arxiv.org/pdf/2209.05433.pdf>_
    F8_E4M3,
    // F8_E8M0 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
    F8_E8M0,
    // FP8 E4M3 (FNUZ) <https://arxiv.org/pdf/2206.02915.pdf>_
    F8_E4M3FNUZ,
    // FP8 E5M2 (FNUZ) <https://arxiv.org/pdf/2206.02915.pdf>_
    F8_E5M2FNUZ,
    // Signed integer (16-bit)
    I16,
    // Unsigned integer (16-bit)
    U16,
    // Half-precision floating point
    F16,
    // Brain floating point
    BF16,
    // Signed integer (32-bit)
    I32,
    // Unsigned integer (32-bit)
    U32,
    // Floating point (32-bit)
    F32,
    // Complex (32-bit parts)
    C64,
    // Floating point (64-bit)
    F64,
    // Signed integer (64-bit)
    I64,
    // Unsigned integer (64-bit)
    U64,
    // Unsigned integer (4bits)
    F4_E2M1,
};
