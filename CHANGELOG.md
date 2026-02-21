# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- `forest_plot(save=...)` now uses figure-bound saving (`fig.savefig`) instead of global pyplot state (`plt.savefig`), preventing wrong-figure saves in multi-figure/notebook contexts.
- Save paths are normalized with `os.fspath(...)` before writing.
- Parent directories for `save` targets are created automatically when missing.

### Added
- Smoke-test coverage for save behavior:
  - verifies `Figure.savefig` is used (and `plt.savefig` is not),
  - verifies nested parent-directory creation for save paths.

## [1.0.1] - 2026-02-20

### Fixed
- Patch release for bugfixes in `forest_plot()` behavior discovered after `1.0.0`.

### Changed
- Documentation updates for API/docs sync and release metadata guidance.

## [1.0.0] - 2026-02-19

### Added
- Initial public release of `forestplotx`.
- `forest_plot()` API for publication-style table + forest plot rendering.
- Support for `binom`, `gamma`, `linear`, and `ordinal` model outputs.
- Input normalization with link-aware exponentiation policy.
- Deterministic 4-case layout presets (general stats on/off x one/two outcomes).
- Axis configuration utilities including log/linear handling and optional outlier clipping.
- Test suite and CI workflow for structural/behavioral validation.
