import os, numpy as np, platform
from setuptools import setup, Extension
from Cython.Build import cythonize

# Package name
__package__ = "sqlcycli"


# Create Extension
def extension(src: str, include_np: bool, *extra_compile_args: str) -> Extension:
    # Prep name
    if "/" in src:
        folders: list[str] = src.split("/")
        file: str = folders.pop(-1)
    else:
        folders: list[str] = []
        file: str = src
    if "." in file:  # . remove extension
        file = file.split(".")[0]
    name = ".".join([__package__, *folders, file])

    # Prep source
    if "/" in src:
        file = src.split("/")[-1]
    else:
        file = src
    source = os.path.join("src", __package__, *folders, file)

    # Extra arguments
    extra_args = list(extra_compile_args) if extra_compile_args else None

    # Create extension
    if include_np:
        return Extension(
            name,
            [source],
            extra_compile_args=extra_args,
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        )
    else:
        return Extension(name, [source], extra_compile_args=extra_args)


# Build Extensions
if platform.system() == "Windows":
    extensions = [
        # fmt: off
        extension("aio/connection.py", True),
        extension("aio/pool.py", True),
        extension("constants/_CLIENT.py", False),
        extension("constants/_COMMAND.py", False),
        extension("constants/_FIELD_TYPE.py", False),
        extension("constants/_SERVER_STATUS.py", False),
        extension("_auth.py", True),
        extension("_connect.py", False),
        extension("_optionfile.py", False),
        extension("_ssl.py", False),
        extension("charset.py", False),
        extension("connection.py", True),
        extension("errors.py", True),
        extension("protocol.py", True),
        extension("sqlfunc.py", True),
        extension("sqlintvl.py", True),
        extension("transcode.py", True),
        extension("typeref.py", False),
        extension("utils.py", True),
        # fmt: on
    ]
else:
    extensions = [
        # fmt: off
        extension("aio/connection.py", True, "-Wno-unreachable-code", "-Wno-incompatible-pointer-types", "-Wno-sign-compare"),
        extension("aio/pool.py", True, "-Wno-unreachable-code", "-Wno-incompatible-pointer-types"),
        extension("constants/_CLIENT.py", False),
        extension("constants/_COMMAND.py", False),
        extension("constants/_FIELD_TYPE.py", False),
        extension("constants/_SERVER_STATUS.py", False),
        extension("_auth.py", True, "-Wno-unreachable-code"),
        extension("_connect.py", False, "-Wno-unreachable-code", "-Wno-incompatible-pointer-types"),
        extension("_optionfile.py", False, "-Wno-unreachable-code"),
        extension("_ssl.py", False, "-Wno-unreachable-code"),
        extension("charset.py", False, "-Wno-unreachable-code"),
        extension("connection.py", True, "-Wno-unreachable-code", "-Wno-sign-compare"),
        extension("errors.py", True, "-Wno-unreachable-code"),
        extension("protocol.py", True, "-Wno-unreachable-code"),
        extension("sqlfunc.py", True, "-Wno-unreachable-code"),
        extension("sqlintvl.py", True, "-Wno-unreachable-code"),
        extension("transcode.py", True, "-Wno-unreachable-code"),
        extension("typeref.py", False),
        extension("utils.py", True, "-Wno-unreachable-code"),
        # fmt: on
    ]

# Build
setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
        annotate=True,
    ),
)
