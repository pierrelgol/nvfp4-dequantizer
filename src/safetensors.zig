const std = @import("std");

const TensorMap = std.StringArrayHashMapUnmanaged;

pub const Tensor = struct {
    pub const MetadataMap = std.StringArrayHashMapUnmanaged;
    pub const Info = struct {};
    pub const Shape = []u64;
    pub const Dtype = enum {};
    pub const DataOffsets = [2]u64;
};
