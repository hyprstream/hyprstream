# amd64 release runner capability evidence (2026-10-20)

The [dedicated runner capability probe](https://github.com/hyprstream/hyprstream/actions/runs/39519303291)
passed on PR #1759 head `f76b477426d20f8755dee88e615a0ea0817daafe`
with `runner_label=hyprstream-release`. Its
[job](https://github.com/hyprstream/hyprstream/actions/runs/39519303291/job/118958955142)
reported `x86_64`, eight CPUs, 64,163,840 KiB of memory, a dedicated
`/mnt/hypr-ci-cache` mount, Podman 5.8.2, and GHCR reachability (HTTP 401
authentication challenge). The probe records Podman's presence and the cache
mount; it does not build an image.

[OCI Runtime Image Validation run 39443276226](https://github.com/hyprstream/hyprstream/actions/runs/39443276226)
passed on the same exact head. Its
[amd64 job](https://github.com/hyprstream/hyprstream/actions/runs/39443276226/job/118698706224)
ran on `hyprstream-amd64-release-i-05a67a07025d3addb` and passed the
`x86_64` runner check, the dedicated `/mnt/hypr-ci-cache` mount check, Podman
5.8.2 image build, PostgreSQL qualification, runtime smoke test, and nonroot
UID assertions. The [arm64 job](https://github.com/hyprstream/hyprstream/actions/runs/39443276226/job/118698705610)
and [aggregate gate](https://github.com/hyprstream/hyprstream/actions/runs/39443276226/job/118737532894)
also passed. Together these runs prove the release runner class accepts jobs
and supplies the capabilities required by the OCI gate at that revision.

The `hyprstream-canary` class has not been probed and remains `unproven` in
`build-matrix.json`. No evidence here establishes its availability.
