from conan import ConanFile


class VextBenchmarksConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        self.requires("benchmark/1.9.5")
        self.requires("eigen/5.0.1")
