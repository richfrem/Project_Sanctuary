try:
    import llama_cpp
    print('llama_cpp import OK')
except Exception as e:
    print('llama-cpp-python import failed:', e)
    raise
