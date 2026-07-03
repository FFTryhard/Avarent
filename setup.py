import os
import sys
import subprocess
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Convert to forward slashes for CMake to avoid Windows trailing backslash escaping bugs
        extdir = extdir.replace(os.sep, "/")
            
        import pybind11
        pybind11_dir = pybind11.get_cmake_dir().replace(os.sep, "/")
        
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11_dir}",
            "-DCMAKE_BUILD_TYPE=Release"
        ]
        
        build_args = []
        
        if sys.platform.startswith("win"):
            cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={extdir}"]
            build_args += ["--config", "Release"]

        env = os.environ.copy()
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp, env=env
        )

setup(
    name="avarent",
    version="0.1.0",
    description="Avarent Quantitative Intelligence Engine",
    author="Avarent",
    packages=find_packages(), 
    # This maps the 'mga_module' name to the 'cpp/' folder containing CMakeLists.txt
    ext_modules=[CMakeExtension("mga_module", sourcedir="cpp")],
    cmdclass={"build_ext": CMakeBuild},
    install_requires=[
        "pandas",
        "numpy",
        "fastapi",
        "uvicorn",
        "python-dotenv",
        "slowapi",
        "apscheduler",
        "yfinance",
    ],
    zip_safe=False,
)
