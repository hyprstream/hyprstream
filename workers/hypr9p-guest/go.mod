module github.com/hyprstream/hyprstream/workers/hypr9p-guest

go 1.26

require (
	github.com/hanwen/go-fuse/v2 v2.8.0
	github.com/hugelgupf/p9 v0.3.1-0.20240118043522-6f4f11e5296e
	golang.org/x/sys v0.43.0
)

require github.com/u-root/uio v0.0.0-20240224005618-d2acac8f3701 // indirect

// hyprstream's 9P server + wanix-guest both use the progrium p9 fork. The
// upstream fork (module path github.com/hugelgupf/p9) does not yet expose the
// standard Tattach.uname field, so both Go consumers pin the AttachUname fork
// at github.com/hyprstream/p9 (same module path, progrium/p9 lineage). Kept in
// lockstep with workers/wanix-guest/go.mod. Upstream: https://github.com/progrium/p9/pull/2
replace github.com/hugelgupf/p9 => github.com/hyprstream/p9 v0.0.0-20260714231754-60421d6579fe
