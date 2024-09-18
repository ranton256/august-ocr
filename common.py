import os


def get_preproc_path(path, output_dir):
    path_base, path_ext = os.path.splitext(os.path.basename(path))
    preproc_file_name = path_base + "_proc.jpg"
    return os.path.join(output_dir, preproc_file_name)
