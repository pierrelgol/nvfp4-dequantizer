const std = @import("std");

pub const Benchmark = struct {
    started: std.Io.Timestamp,
    elapsed_ns: u64 = 0,
    bytes: u64 = 0,

    pub fn start(io: std.Io) Benchmark {
        return .{ .started = std.Io.Clock.now(.awake, io) };
    }

    pub fn stop(self: *Benchmark, io: std.Io, bytes: ?u64) void {
        self.elapsed_ns = @intCast(self.started.untilNow(io, .awake).nanoseconds);
        self.bytes = bytes orelse 0;
    }

    pub fn format(self: Benchmark, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        const throughput = if (self.elapsed_ns == 0)
            0
        else
            self.bytes * std.time.ns_per_s / self.elapsed_ns;

        try writer.print("Duration: {f}, throughput: {Bi}/s", .{
            std.Io.Duration.fromNanoseconds(self.elapsed_ns),
            throughput,
        });
    }
};
