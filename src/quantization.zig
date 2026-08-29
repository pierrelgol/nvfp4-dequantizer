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
