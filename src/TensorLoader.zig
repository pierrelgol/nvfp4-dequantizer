const std = @import("std");
const mem = std.mem;
const heap = std.heap;
const math = std.math;
const fmt = std.fmt;
const log = std.log;
const ascii = std.ascii;
const Io = std.Io;
const process = std.process;
pub const Safetensors = @import("Safetensors.zig");
pub const TensorLoader = @This();
pub const json = std.json;

input: *Io.Queue(Safetensors.TensorMetaData),
output: *Io.Queue(),
