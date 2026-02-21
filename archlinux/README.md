# Arch Linux 

This is for building and installing a package in arch linux.

## Prerequisites

You need to be running [Arch Linux](https://archlinux.org/) and have the `base-devel` installed as bare mininum.

Any other dependencies can be seen by running the following:

```bash
grep -niHe "depends" PKGBUILD 
```

## Installation

Assuming all the dependencies are met, simply type `makepkg -si`.
