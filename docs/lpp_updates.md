# Overview
Here I outline the steps I took to update the SECDA LLM to the latest version of the llama.cpp codebase. This is a work in progress, and I will update this document as I make progress.


# Steps Taken

1. I cloned the latest version of the llama.cpp codebase from GitHub.
2. I copied ggml-secda folder from my previous version of the codebase into /ggml/src/ggml-secda in the new codebase.
3. I copied the `ggml-secda.h` and `ggml-secda.cpp` files from my previous version of the codebase into the new codebase.
4. I added "acc_del" into ggml-secda folder
5. I update cmakelist.txt file in ggml/src/, ggml/ and ggml/src/ggml-secda to include the new files and directories for the SECDA backend.
6. I also updated cmakelist.txt file in acc_del directory.
7. I update ggml-secda.cpp and ggml-secda.h file to include the new code to adpat the SECDA backend to the latest version of the llama.cpp codebase. This involved updating the function signatures and adding new functions as needed to match the latest API of the llama.cpp codebase.
8. I edited the `ggml-backend-reg.cpp` file to include the new SECDA backend registration code.
9. I also added FindSECDA_TOOLS.cmake and FindSYSC.cmake files to the cmake directory to help with finding the necessary tools for building the SECDA backend.
10. I updated the CMakePresets.json file to include the new CMake options for building the SECDA backend.