const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .optimize = optimize,
        .target = target,
        .imports = &.{},
    });

    const exe = b.addExecutable(.{
        .name = "nvfp4_dequantizer",
        .root_module = exe_mod,
    });

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.addFileArg2(b.path("models/gemma-3-270m-it-NVFP4/model.safetensors"), .{ .make_absolute = true });
    run_cmd.addPassthruArgs();
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());

    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });

    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_exe_tests.step);

    const exe_check = b.addExecutable(.{
        .name = "nvfp4_dequantizer",
        .root_module = exe_mod,
    });

    const check_step = b.step("check", "Run checks");
    check_step.dependOn(&exe_check.step);
}
