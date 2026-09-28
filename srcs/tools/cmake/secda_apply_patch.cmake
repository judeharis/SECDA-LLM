# cmake -DPATCH=<patch exe> -DIN=<upstream file> -DPATCHFILE=<patch> -DOUT=<output> -P secda_apply_patch.cmake
# Writes OUT only if the patch applies cleanly. A patch that does not apply, or
# that looks already applied (--forward: never reverse-applied), is an error,
# and no patched copy (old or partial) is left behind.
execute_process(
  COMMAND "${PATCH}" --force --forward --fuzz=0 --no-backup-if-mismatch --reject-file=- -o "${OUT}.tmp" "${IN}" "${PATCHFILE}"
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  file(REMOVE "${OUT}.tmp" "${OUT}")
  message(FATAL_ERROR "${PATCHFILE} does not apply to ${IN} (the pinned llama.cpp); "
                      "see srcs/tools/refresh_patch.sh.\n${out}${err}")
endif()
file(RENAME "${OUT}.tmp" "${OUT}")
