const std = @import("std");

pub const Timer = struct {
    name: []const u8,
    started: std.Io.Timestamp,
    elapsed_ns: u64 = 0,
    bytes: ?u64 = null,

    pub fn start(name: []const u8, io: std.Io) Timer {
        return .{
            .name = name,
            .started = std.Io.Clock.now(.awake, io),
        };
    }

    pub fn stop(self: *Timer, io: std.Io, bytes: ?u64) void {
        self.elapsed_ns = @intCast(self.started.untilNow(io, .awake).nanoseconds);
        self.bytes = bytes;
        std.debug.print("{f}\n", .{self.*});
    }

    pub fn format(self: Timer, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print("{s}: {f}", .{
            self.name,
            std.Io.Duration.fromNanoseconds(self.elapsed_ns),
        });

        if (self.bytes) |bytes| {
            const seconds = @as(f64, @floatFromInt(self.elapsed_ns)) / std.time.ns_per_s;
            const bytes_per_second = if (seconds == 0)
                0
            else
                @as(f64, @floatFromInt(bytes)) / seconds;

            try writer.print(", {Bi}, {d:.2} MiB/s", .{
                bytes,
                bytes_per_second / (1024 * 1024),
            });
        }
    }
};
