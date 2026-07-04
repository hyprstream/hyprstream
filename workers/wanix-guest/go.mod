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

// wanix's p9kit imports the path github.com/hugelgupf/p9 but requires the
// progrium fork. Go does NOT apply a dependency's replace directives
// transitively, so this MUST be duplicated here verbatim from wanix's go.mod.
replace github.com/hugelgupf/p9 => github.com/progrium/p9 v0.0.0-20260529042029-b49ec572080f
