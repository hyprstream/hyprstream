## p9

[![CircleCI](https://circleci.com/gh/hugelgupf/p9.svg?style=svg)](https://circleci.com/gh/hugelgupf/p9)
[![Go Report Card](https://goreportcard.com/badge/github.com/hugelgupf/p9)](https://goreportcard.com/report/github.com/hugelgupf/p9)
[![GoDoc](https://godoc.org/github.com/hugelgupf/p9?status.svg)](https://godoc.org/github.com/hugelgupf/p9)

p9 is a Golang 9P2000.L client and server originally written for gVisor. p9
supports Windows, BSD, and Linux on most Go-available architectures.

### Server Example

For how to start a server given a `p9.Attacher` implementation, see
[cmd/p9ufs](cmd/p9ufs/p9ufs.go).

For how to implement a `p9.Attacher` and `p9.File`, see as an example
[staticfs](fsimpl/staticfs/staticfs.go), a simple static file system.
Boilerplate templates for `p9.File` implementations are in
[templatefs](fsimpl/templatefs/).

A test suite for server-side `p9.Attacher` and `p9.File` implementations is
being built at [fsimpl/test](fsimpl/test/filetest.go).

### Client Example

```go
import (
    "log"
    "net"

    "github.com/hugelgupf/p9/p9"
)

func main() {
  conn, err := net.Dial("tcp", "localhost:8000")
  if err != nil {
    log.Fatal(err)
  }

  // conn can be any net.Conn.
  client, err := p9.NewClient(conn)
  if err != nil {
    log.Fatal(err)
  }

  // root will be a p9.File and supports all those operations.
  root, err := client.Attach("/")
  if err != nil {
    log.Fatal(err)
  }

  // For example:
  _, _, attrs, err := root.GetAttr(p9.AttrMaskAll)
  if err != nil {
    log.Fatal(err)
  }

  log.Printf("Attrs of /: %v", attrs)
}
```
