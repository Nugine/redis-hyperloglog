dev:
    clang-format -i cpp/*.cpp
    cargo fmt
    cargo clippy
    cargo test
    cargo miri test
