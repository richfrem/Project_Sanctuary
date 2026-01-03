try:
    import xformers
    print('xformers import OK; version =', getattr(xformers, '__version__', 'unknown'))
except Exception as e:
    print('xformers import failed:', e)
    raise
