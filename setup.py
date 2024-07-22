import os, numpy as np, platform
from setuptools import setup, Extension
from Cython.Build import cythonize

# Package name
__package__ = "sqlcycli"


# Create Extension
def extension(filename: str, include_np: bool, *extra_compile_args: str) -> Extension:
    # Extra arguments
    extra_args = list(extra_compile_args) if extra_compile_args else None
    # Name
    name: str = "%s.%s" % (__package__, filename.split(".")[0])
    source: str = os.path.join("src", __package__, filename)
    # Create extension
    if include_np:
        return Extension(
            name,
            sources=[source],
            extra_compile_args=extra_args,
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        )
    else:
        return Extension(name, sources=[source], extra_compile_args=extra_args)


# Create Constant Extension
def folder_extension(
    folder: str,
    filename: str,
    include_np: bool,
    *extra_compile_args: str,
) -> Extension:
    # Extra arguments
    extra_args = list(extra_compile_args) if extra_compile_args else None
    # Name
    name: str = "%s.%s.%s" % (__package__, folder, filename.split(".")[0])
    source: str = os.path.join("src", __package__, folder, filename)
    # Create extension
    if include_np:
        return Extension(
            name,
            sources=[source],
            extra_compile_args=extra_args,
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        )
    else:
        return Extension(name, sources=[source], extra_compile_args=extra_args)


# Build Extensions
if platform.system() == "Windows":
    extensions = [
        folder_extension("aio", "connection.py", True),
        folder_extension("aio", "pool.py", True),
        folder_extension("constants", "_CLIENT.py", False),
        folder_extension("constants", "_COMMAND.py", False),
        folder_extension("constants", "_FIELD_TYPE.py", False),
        folder_extension("constants", "_SERVER_STATUS.py", False),
        extension("_auth.py", True),
        extension("_optionfile.py", False),
        extension("_ssl.py", False),
        extension("charset.py", False),
        extension("connection.py", True),
        extension("errors.py", False),
        extension("protocol.py", True),
        extension("transcode.py", True),
        extension("typeref.py", False),
    ]
else:
    extensions = [
        folder_extension(
            "aio",
            "connection.py",
            True,
            "-Wno-unreachable-code",
            "-Wno-incompatible-pointer-types",
        ),
        folder_extension(
            "aio",
            "pool.py",
            True,
            "-Wno-unreachable-code",
            "-Wno-incompatible-pointer-types",
        ),
        folder_extension("constants", "_CLIENT.py", False),
        folder_extension("constants", "_COMMAND.py", False),
        folder_extension("constants", "_FIELD_TYPE.py", False),
        folder_extension("constants", "_SERVER_STATUS.py", False),
        extension("_auth.py", True, "-Wno-unreachable-code"),
        extension("_optionfile.py", False, "-Wno-unreachable-code"),
        extension("_ssl.py", False, "-Wno-unreachable-code"),
        extension("charset.py", False, "-Wno-unreachable-code"),
        extension("connection.py", True, "-Wno-unreachable-code"),
        extension("errors.py", False),
        extension("protocol.py", True, "-Wno-unreachable-code"),
        extension("transcode.py", True, "-Wno-unreachable-code"),
        extension("typeref.py", False),
    ]

# Build
setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
        annotate=True,
    ),
)
