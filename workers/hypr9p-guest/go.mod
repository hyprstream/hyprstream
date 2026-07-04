module github.com/hyprstream/hyprstream/workers/hypr9p-guest

go 1.26

require (
	github.com/hugelgupf/p9 v0.3.1-0.20240118043522-6f4f11e5296e
	golang.org/x/sys v0.43.0
)

require github.com/u-root/uio v0.0.0-20240224005618-d2acac8f3701 // indirect

// hyprstream's 9P server + wanix-guest both use the progrium p9 fork; pin the
// same one here (Go does not apply a dependency's replace directives, so this is
// declared, not inherited). Kept in lockstep with workers/wanix-guest/go.mod.
replace github.com/hugelgupf/p9 => github.com/progrium/p9 v0.0.0-20260529042029-b49ec572080f
