import pytest
import sys
import os

def main():
    print("======================================================================")
    print("                Executing Biomolecular Design E2E Test Suite          ")
    print("======================================================================")
    
    test_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "tests/test_e2e_suite.py"))
    
    # Run pytest.main with the test file and verbose output
    exit_code = pytest.main(["-v", test_file])
    
    print("\n======================================================================")
    if exit_code == 0:
        # Don't restate a case count here -- it drifted (this said 49 while the
        # suite had grown well past it). pytest's own summary above is the
        # authoritative count.
        print("          SUCCESS: the E2E suite passed. See the pytest summary above.")
    else:
        print(f"          FAILURE: Pytest execution exited with code {exit_code}        ")
    print("======================================================================")
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
