# hyprstream-csi

CSI node plugin for `csi.hyprstream.io` (#790).

The node plugin requests `/oauth/mount-ticket` using the pod-projected service
account token supplied by CSI `podInfoOnMount`, then mounts the scoped 9P tree at
the kubelet target path.

Mounters:

- `mounter=fuse`: starts `hypr9p-guest --dial <target> --fuse-mount <target>`
  with the minted ticket in `HYPRSTREAM_9P_UNAME`, so the ticket is presented as
  9P `uname` and never appears in process argv.
- `mounter=kernel`: calls the node-local stream bridge for kernel v9fs
  `trans=fd`; the bridge terminates the selected carrier and performs the mount
  with the ticket as `uname=`.

The transport is a dial-time carrier. The Phase-1 FUSE client supports raw
`vsock`, `unix`, and `tcp` dial targets. `webtransport` / iroh-QUIC is the
cross-node target once the node bridge/dialer lands; UDS is only for co-located
nodes.
