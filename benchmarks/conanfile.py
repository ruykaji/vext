from conan import ConanFile


class VextBenchmarksConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        self.requires("abseil/20240722.0")
        self.requires("benchmark/1.9.5")
        self.requires("eigen/3.4.1")
