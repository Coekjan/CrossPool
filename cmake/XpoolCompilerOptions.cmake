include_guard(GLOBAL)

add_library(xpool_compiler_options INTERFACE)
target_compile_options(xpool_compiler_options INTERFACE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra>
  $<$<AND:$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>,$<BOOL:${XPOOL_WARNINGS_AS_ERRORS}>>:-Werror>
  $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=-Wall>
  $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=-Wextra>
  $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--Wreorder>
  $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--Wdefault-stream-launch>
  $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--Wext-lambda-captures-this>
  $<$<AND:$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>,$<BOOL:${XPOOL_WARNINGS_AS_ERRORS}>>:-Xcompiler=-Werror>
  $<$<AND:$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>,$<BOOL:${XPOOL_WARNINGS_AS_ERRORS}>>:--Werror=all-warnings,cross-execution-space-call>
)
