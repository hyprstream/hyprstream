#!/usr/bin/env tclsh
# @description List files in a directory
# @param path:string:required Directory path to list
# @param pattern:string:optional Glob pattern to filter files (defaults to '*')

# Check if required parameter exists
if {![info exists path]} {
    error "Missing required parameter: path"
}

# Default pattern to * if not provided
if {![info exists pattern]} {
    set pattern "*"
}

# List files matching the pattern in the directory
set files [glob -nocomplain -directory $path $pattern]

# Sort and display the files
foreach file [lsort $files] {
    puts $file
}