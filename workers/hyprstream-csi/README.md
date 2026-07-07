# hyprstream-csi

CSI node plugin for `csi.hyprstream.io` (#790).

The node plugin requests `/oauth/mount-ticket` using the pod-projected service
account token supplied by CSI `podInfoOnMount`, then attaches the selected 9P
export at the kubelet target path. CSI volume attributes use `aname` (or
`exportRef`) as the Plan 9 attach selector; `namespacePath` is intentionally
rejected so Kubernetes does not define a private hyprstream path convention.

Mounters:

- `mounter=fuse`: starts `hypr9p-guest --dial <target> --fuse-mount <target>`
  with the minted ticket in `HYPRSTREAM_9P_UNAME`, so the ticket is presented as
  9P `uname` and never appears in process argv. The export selector is passed as
  9P `aname`.
- `mounter=kernel`: calls the node-local stream bridge for kernel v9fs
  `trans=fd`; the bridge terminates the selected carrier and performs the mount
  with the ticket as `uname=` and the export selector as `aname`.

The transport is a dial-time carrier. The Phase-1 FUSE client supports raw
`vsock`, `unix`, and `tcp` dial targets, and requires an operator-provided
node-local listener/bridge endpoint. `webtransport` / iroh-QUIC is the
cross-node target once the node bridge/dialer lands; UDS is only for co-located
nodes. The kernel mounter is opt-in until the `hyprstream-csi-bridge` binary
ships.
