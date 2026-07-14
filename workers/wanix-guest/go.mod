module github.com/hyprstream/hyprstream/workers/wanix-guest

go 1.26

require (
	github.com/hugelgupf/p9 v0.3.1-0.20240118043522-6f4f11e5296e
	tractor.dev/wanix v0.0.0-20260703022758-381150790444
)

require (
	github.com/creack/pty v1.1.24 // indirect
	github.com/u-root/uio v0.0.0-20240224005618-d2acac8f3701 // indirect
	golang.org/x/sys v0.43.0 // indirect
	tractor.dev/toolkit-go v0.0.0-20250103001615-9a6753936c19 // indirect
)

// wanix's p9kit imports the path github.com/hugelgupf/p9. The upstream progrium
// fork does not yet expose the standard Tattach.uname field, so both Go
// consumers pin the AttachUname fork at github.com/hyprstream/p9 (same module
// path, progrium/p9 lineage). Kept in lockstep with workers/hypr9p-guest/go.mod.
// Upstream: https://github.com/progrium/p9/pull/2
replace github.com/hugelgupf/p9 => github.com/hyprstream/p9 v0.0.0-20260714225611-9155f405ff22
