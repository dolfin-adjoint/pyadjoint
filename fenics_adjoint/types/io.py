import backend
from pyadjoint.tape import annotate_tape


__HDF5File_read__ = backend.HDF5File.read


def HDF5File_read(self, *args, **kwargs):
    annotate = annotate_tape(kwargs)
    output = __HDF5File_read__(self, *args, **kwargs)

    if annotate:
        func = args[0]
        func.create_block_variable()
    return output

backend.HDF5File.read = HDF5File_read
