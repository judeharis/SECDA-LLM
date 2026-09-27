load("@rules_cc//cc:defs.bzl", "cc_library")
load("//:def.bzl", "secda_copts")

package(default_visibility = ["//visibility:public"])

common_copts = secda_copts()

cc_library(
    name = "accelerator",
    srcs = [
        "acc_config.sc.h",
    ] + select({
        "@secda_core//:sysc": ["acc.sc.cc"],
        "//conditions:default": [],
    }),
    hdrs = select({
        "@secda_core//:sysc": [
            "acc.sc.h",
            "acc_control_unit.sc.h",
            "acc_load_unit.sc.h",
            "acc_scheduler.sc.h",
            "acc_softmax_unit.sc.h",
            "acc_store_unit.sc.h",
            "bfp_processor_compute.sc.h",
            "bfp_processor_controller.sc.h",
            "bfp_processor_functions.sc.h",
            "bfp_processor_threads.sc.h",
            "bfp_processor_top.sc.h",
        ],
        "//conditions:default": [],
    }),
    copts = common_copts,
    deps = ["//third_party:secdav5"],
)
