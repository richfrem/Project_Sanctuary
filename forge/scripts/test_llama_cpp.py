try:
    import llama_cpp
    print('llama_cpp import OK')

    # Verify CUDA support in the bridge
    cuda_supported = llama_cpp.llama_supports_gpu_offload()
    print(f'llama-cpp-python CUDA support: {cuda_supported}')
    if not cuda_supported:
        raise RuntimeError('llama-cpp-python was not built with CUDA support. Re-run the CMAKE_ARGS installation command.')

except Exception as e:
    print('llama-cpp-python test failed:', e)
    raise
