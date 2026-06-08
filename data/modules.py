import os

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

def is_image_pair(inp_name, tar_name):
    """
    Check if the input and target image names form a pair based on naming convention.
    Match files based on both prefix (e.g., 'pie-' or 'rain-') and numeric suffix.
    """
    inp_prefix, inp_core = inp_name.split('-', 1)
    tar_prefix, tar_core = tar_name.split('-', 1)
    if inp_prefix == "pie" and tar_prefix == "pie":
        return inp_core.replace("rain-", "") == tar_core.replace("norain-", "")
    elif inp_prefix == "rain" and tar_prefix == "norain":
        return inp_core.replace("rain-", "") == tar_core.replace("norain-", "")
    return False