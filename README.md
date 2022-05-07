# IPNIS

## Examples

```bash
cargo run --release --package ipnis-modules-{module}-example
```

Belows are available modules:
(TODO: downloading the .onnx models are not supported yet!)

### NLP

* question-answering

### Vision

* image-classification

## License

* IPNIS Modules (`ipnis-modules-*`) and all other utilities are licensed under [Apache 2.0](LICENSE-APACHE2).
* IPNIS Runtime (`/runtime/*` / `ipnis-runtime`) is licensed under [GPL v3.0 with a classpath linking exception](LICENSE-GPL3).

The reason for the split-licensing is to ensure that for the vast majority of teams using IPNIS to create feature-chains, then all changes can be made entirely in Apache2-licensed code, allowing teams full freedom over what and how they release and giving licensing clarity to commercial teams.
