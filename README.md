# nvfp4-dequantizer

Streaming NVFP4 to F32 converter for Safetensors files, written in Zig.

## Architecture

The input header is parsed and its tensors are classified. A new Safetensors header is built for the F32 output. Tensor data then moves through a bounded pipeline with four dequantization workers and an ordered writer; tensors that do not need conversion are copied unchanged.

## Benchmark

```text
safetensor parsing: 349.611us, 67.46875KiB, 188.46 MiB/s
build output header: 80.176us
write output header: 151.706us, 38.9140625KiB, 250.50 MiB/s
dequantize: 181.372ms, 1022.6071701049805MiB, 5638.17 MiB/s
```

## SIMD dequantization

The SIMD dequantization kernel is one of the gems of this project. It is a Zig port and adaptation of the NVFP4 decoding used by llama.cpp's NVFP4 × Q8 dot-product kernel. This version keeps the unpacking, lookup, and scaling stages, but writes sixteen F32 weights instead of continuing into a dot product. It expands one 8-byte NVFP4 block while keeping the implementation compact and readable.

Each input byte contains two E2M1 values. The low nibble is the first weight and the high nibble is the second. One UE4M3 scale is shared by the block, along with the tensor's inverse global scale.

```zig
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
                break :blk @"llvm.x86.ssse3.pshuf.b.128"(table, indices);
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
```

Line by line:

1. `PackedVector` treats the eight packed bytes as eight SIMD lanes.
2. `CodeVector` holds the sixteen unpacked 4-bit codes.
3. `SignedVector` holds sixteen signed values read from the E2M1 lookup table.
4. `FloatVector` holds sixteen F32 values.
5. `ShuffleMask` is the index type used by `@shuffle`.
6. `nibble_mask` repeats `0x0f` across all eight lanes so the low nibble can be isolated in parallel.
7. `nibble_shift` repeats `4`, the number of bits used to move each high nibble down.
8. `interleave_mask` asks `@shuffle` for `low[0], high[0], low[1], high[1]` and so on. Negative indices select the second input vector: `-1` selects `high[0]`, `-2` selects `high[1]`.
9. `quantized_weights` reinterprets the input array as the eight-lane vector without changing its bits.
10. `low` extracts all eight low nibbles with one vector AND.
11. `high` shifts all eight high nibbles into the range `0...15`.
12. `code` interleaves both halves, restoring the original order of the sixteen weights.
13. `code_values` provides an array view for the scalar fallback.
14. `table` reinterprets the sixteen-entry E2M1 lookup table as a byte vector.
15. `indices` gives the lookup instructions sixteen table indices.
16. The architecture switch chooses the available byte-table lookup implementation.
17. On x86-64 with SSSE3, `pshufb` performs all sixteen E2M1 lookups in one instruction.
18. Older x86-64 targets use the small scalar loop to read the same lookup table.
19. AArch64 uses NEON `tbl`, also performing sixteen byte lookups at once.
20. Other architectures use the same scalar lookup loop.
21. `@floatFromInt` converts all sixteen signed lookup results to F32 lanes.
22. `unsigned_values` exposes their IEEE 754 bits so individual bit patterns can be repaired.
23. `negative_zero_code` identifies E2M1 code `1000`, which represents negative zero.
24. `negative_zero_bits` contains the IEEE 754 F32 representation of `-0.0`.
25. `@select` keeps normal converted values and replaces code `1000` with true negative zero.
26. `scaling_factor` decodes the block's UE4M3 scale and applies the inverse global scale.
27. `scaling_vector` broadcasts that scalar to all sixteen lanes.
28. `decoded` multiplies all weights by the shared scale in parallel.
29. The final `@bitCast` returns the vector as `[16]f32`.
