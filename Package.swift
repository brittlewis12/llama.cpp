// swift-tools-version:5.9

import PackageDescription

let package = Package(
    name: "llama",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v1),
        .watchOS(.v4), // ???
        .tvOS(.v14)
    ],
    products: [
        .library(name: "llama", targets: ["llama"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ggerganov/ggml.git", .branch("release"))
    ],
    targets: [
        .target(
            name: "llama",
            dependencies: ["ggml"],
            path: ".",
            exclude: ["ggml-metal.metal"],
            sources: [
                "llama.cpp",
            ],
            publicHeadersPath: "spm-headers",
            cSettings: [
                // NOTE: NEW_LAPACK will required iOS version 16.4+
                // We should consider add this in the future when we drop support for iOS 14
                // (ref: ref: https://developer.apple.com/documentation/accelerate/1513264-cblas_sgemm?language=objc)
                .define("ACCELERATE_NEW_LAPACK"),
                .define("ACCELERATE_LAPACK_ILP64")
                .unsafeFlags(["-Wno-shorten-64-to-32", "-O3", "-DNDEBUG"]),
                .define("GGML_USE_ACCELERATE"),
                .unsafeFlags(["-fno-objc-arc"]),
                .define("GGML_USE_METAL"),
            ],
            linkerSettings: [
                .linkedFramework("Accelerate")
            ]
        )
    ],
    cxxLanguageStandard: .cxx11
)
