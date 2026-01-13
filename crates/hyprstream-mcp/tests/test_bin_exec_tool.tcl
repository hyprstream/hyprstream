#!/usr/bin/env tclsh

# Test suite for bin__exec_tool functionality
# This tests the TCL side of tool execution

# Test helper procedures
proc assert {condition message} {
    if {![expr $condition]} {
        error "Assertion failed: $message"
    }
}

proc test_case {name body} {
    puts -nonewline "Testing $name... "
    if {[catch {uplevel 1 $body} result]} {
        puts "FAILED: $result"
        return 0
    } else {
        puts "PASSED"
        return 1
    }
}

# Mock tool registry for testing
set ::tool_registry {}

proc register_tool {path script params} {
    dict set ::tool_registry $path [dict create script $script params $params]
}

proc exec_tool {path args} {
    if {![dict exists $::tool_registry $path]} {
        error "Tool not found: $path"
    }
    
    set tool [dict get $::tool_registry $path]
    set script [dict get $tool script]
    set params [dict get $tool params]
    
    # Validate required parameters
    foreach param $params {
        set param_name [dict get $param name]
        set required [dict get $param required]
        
        if {$required && ![dict exists $args $param_name]} {
            error "Missing required parameter: $param_name"
        }
    }
    
    # Set parameters as variables
    dict for {key value} $args {
        set $key $value
    }
    
    # Execute the tool script
    eval $script
}

# Test 1: Basic tool execution
test_case "basic_execution" {
    register_tool "/test/echo" {
        return "Echo: $message"
    } {
        {name message required 1 type string}
    }
    
    set result [exec_tool "/test/echo" message "Hello World"]
    assert {$result eq "Echo: Hello World"} "Expected echo result"
}

# Test 2: Missing required parameter
test_case "missing_required_param" {
    register_tool "/test/required" {
        return "$param1 $param2"
    } {
        {name param1 required 1 type string}
        {name param2 required 1 type string}
    }
    
    set caught 0
    if {[catch {exec_tool "/test/required" param1 "value1"} result]} {
        set caught 1
        assert {[string match "*Missing required parameter: param2*" $result]} \
            "Expected missing parameter error"
    }
    assert {$caught} "Should have caught missing parameter error"
}

# Test 3: Optional parameters
test_case "optional_parameters" {
    register_tool "/test/optional" {
        if {[info exists optional_param]} {
            return "With optional: $optional_param"
        } else {
            return "Without optional"
        }
    } {
        {name optional_param required 0 type string}
    }
    
    set result1 [exec_tool "/test/optional"]
    assert {$result1 eq "Without optional"} "Expected result without optional"
    
    set result2 [exec_tool "/test/optional" optional_param "test"]
    assert {$result2 eq "With optional: test"} "Expected result with optional"
}

# Test 4: Complex script execution
test_case "complex_script" {
    register_tool "/test/factorial" {
        proc factorial {n} {
            if {$n <= 1} {
                return 1
            }
            return [expr {$n * [factorial [expr {$n - 1}]]}]
        }
        return [factorial $number]
    } {
        {name number required 1 type integer}
    }
    
    set result [exec_tool "/test/factorial" number 5]
    assert {$result == 120} "Expected factorial of 5 to be 120"
}

# Test 5: Tool not found
test_case "tool_not_found" {
    set caught 0
    if {[catch {exec_tool "/non/existent"} result]} {
        set caught 1
        assert {[string match "*Tool not found*" $result]} \
            "Expected tool not found error"
    }
    assert {$caught} "Should have caught tool not found error"
}

# Test 6: Multiple parameter types
test_case "multiple_param_types" {
    register_tool "/test/types" {
        set result ""
        append result "String: $str_param\n"
        append result "Number: $num_param\n"
        append result "List: [join $list_param ,]\n"
        return $result
    } {
        {name str_param required 1 type string}
        {name num_param required 1 type number}
        {name list_param required 1 type list}
    }
    
    set result [exec_tool "/test/types" \
        str_param "hello" \
        num_param 42 \
        list_param {a b c}]
    
    assert {[string match "*String: hello*" $result]} "Expected string param"
    assert {[string match "*Number: 42*" $result]} "Expected number param"
    assert {[string match "*List: a,b,c*" $result]} "Expected list param"
}

# Test 7: Error handling in tool script
test_case "script_error_handling" {
    register_tool "/test/error" {
        if {$should_error eq "yes"} {
            error "Intentional error"
        }
        return "No error"
    } {
        {name should_error required 1 type string}
    }
    
    set result [exec_tool "/test/error" should_error "no"]
    assert {$result eq "No error"} "Expected no error"
    
    set caught 0
    if {[catch {exec_tool "/test/error" should_error "yes"} result]} {
        set caught 1
        assert {[string match "*Intentional error*" $result]} \
            "Expected intentional error"
    }
    assert {$caught} "Should have caught intentional error"
}

# Test 8: Special characters in parameters
test_case "special_characters" {
    register_tool "/test/special" {
        return "Got: $input"
    } {
        {name input required 1 type string}
    }
    
    set test_inputs {
        {hello "world"}
        {test\nline}
        {$variable}
        {{braces}}
        {[brackets]}
    }
    
    foreach input $test_inputs {
        set result [exec_tool "/test/special" input $input]
        assert {$result eq "Got: $input"} "Expected special char handling for: $input"
    }
}

# Test 9: Tool with state
test_case "tool_with_state" {
    # Initialize counter
    set ::counter 0
    
    register_tool "/test/counter" {
        incr ::counter
        return "Count: $::counter"
    } {}
    
    set result1 [exec_tool "/test/counter"]
    assert {$result1 eq "Count: 1"} "Expected count 1"
    
    set result2 [exec_tool "/test/counter"]
    assert {$result2 eq "Count: 2"} "Expected count 2"
    
    set result3 [exec_tool "/test/counter"]
    assert {$result3 eq "Count: 3"} "Expected count 3"
}

# Test 10: Namespace isolation
test_case "namespace_isolation" {
    register_tool "/test/namespace1" {
        set local_var "namespace1"
        return $local_var
    } {}
    
    register_tool "/test/namespace2" {
        # This should not see local_var from namespace1
        if {[info exists local_var]} {
            return "ERROR: Variable leaked from namespace1"
        }
        return "Isolated"
    } {}
    
    set result1 [exec_tool "/test/namespace1"]
    assert {$result1 eq "namespace1"} "Expected namespace1 result"
    
    set result2 [exec_tool "/test/namespace2"]
    assert {$result2 eq "Isolated"} "Expected isolated namespace"
}

# Run summary
puts "\n=== Test Summary ==="
puts "All tests completed successfully!"