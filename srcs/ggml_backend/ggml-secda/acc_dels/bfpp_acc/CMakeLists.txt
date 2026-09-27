message(STATUS "===================BFP ACC===================")
# v1/v2 are retired to legacy/ (broken, SECDA-Core axi_support v5 only).
if (SECDA_BFPP_ACC_V1 OR SECDA_BFPP_ACC_V2)
message(FATAL_ERROR "bfpp_acc v1/v2 are retired (acc_dels/bfpp_acc/legacy/); use SECDA_BFPP_ACC_V3 or SECDA_BFPP_ACC_V4")
endif()

if (SECDA_BFPP_ACC_V3)
message(STATUS "Adding bfpp_acc_v3 to build")
add_subdirectory(v3)
endif()

# Jude: Added
if (SECDA_BFPP_ACC_V4)
message(STATUS "Adding bfpp_acc_v4 to build")
add_subdirectory(v4)
endif()
message(STATUS "===================BFP ACC===================")
