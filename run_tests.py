"""
EigenWall Test Runner
Executes all unit tests in tests/ directory and prints a structured summary.
"""

import sys
import traceback

def main():
    print("=" * 60)
    print("Running EigenWall Test Suite")
    print("=" * 60)

    from tests import test_signal_model, test_covariance, test_classical_solver, test_quantum_solver

    tests = [
        ("Signal Model - Steering Vector", test_signal_model.test_steering_vector_properties),
        ("Signal Model - Received Signals", test_signal_model.test_received_signal_generation),
        ("Covariance - Matrix Properties", test_covariance.test_covariance_properties),
        ("Covariance - Regularization", test_covariance.test_regularization),
        ("Classical Solver - Distortionless Constraint", test_classical_solver.test_mvdr_distortionless_constraint),
        ("Classical Solver - Nulling Behavior", test_classical_solver.test_mvdr_nulling_behavior),
        ("Quantum QPE - Golden Matrix R1", test_quantum_solver.test_qpe_golden_matrix),
        ("Quantum HHL - Golden Test A", test_quantum_solver.test_hhl_golden_test_a),
        ("Quantum HHL - Golden Test B", test_quantum_solver.test_hhl_golden_test_b),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n[RUNNING] {name} ...")
        try:
            test_fn()
            print(f"[PASS]    {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL]    {name}: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Summary: {passed} PASSED, {failed} FAILED (Total: {len(tests)})")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
