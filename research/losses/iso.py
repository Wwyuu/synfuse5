def unwrap_hrms(out):
    return out['hrms'] if isinstance(out, dict) else out
