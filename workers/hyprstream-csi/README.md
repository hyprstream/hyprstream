# hyprstream-csi

CSI node plugin for `csi.hyprstream.io` (#790).

The node plugin requests `/oauth/mount-ticket` using the pod-projected service
account token supplied by CSI `podInfoOnMount`, then mounts the scoped 9P tree at
the kubelet target path.

Mounters:

- `mounter=fuse`: starts `hypr9p-guest --dial <carrier> --aname <ticket>
  --fuse-mount <target>`.
- `mounter=kernel`: calls the node-local stream bridge for kernel v9fs
  `trans=fd`; the bridge terminates the selected carrier and performs the mount
  with the ticket as `uname=`.

The transport is a dial-time carrier. `webtransport` / iroh-QUIC is the
cross-node default, `vsock` is for Kata, and UDS is only for co-located nodes.
