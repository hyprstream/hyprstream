#!/usr/bin/env tclsh
# @description Simple hello world tool
# @param name:string:optional Name to greet (defaults to 'World')

# Get the name parameter, default to 'World' if not provided
if {![info exists name]} {
    set name "World"
}

# Print the greeting
puts "Hello, $name!"