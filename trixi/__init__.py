try:
    use_agg = False
    import matplotlib

    if use_agg:
        matplotlib.use("Agg", warn=False)
except Exception:
    print("Failed to import matplotlib")
__version__ = "0.1.2.2"
