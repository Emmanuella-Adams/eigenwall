"""
Measurement, tomography, and post-selection analysis routines for quantum circuits.
"""

from typing import Dict, Any, Tuple
import numpy as np


def sample_and_postselect(
    counts: Dict[str, int],
    n_sys: int,
    total_shots: int
) -> Dict[str, Any]:
    """
    Parse classical measurement register bitstrings and post-select on ancilla = 1.

    In Qiskit, multiple classical registers are typically formatted as:
    "c_sys c_anc" (separated by space) or combined bitstring.

    Returns:
        postselected_counts: Dict mapping system basis state (int or bitstring) to count
        success_probability: Total shots with ancilla=1 / total_shots
        empirical_probabilities: Array of shape (2^n_sys,)
    """
    dim_sys = 2 ** n_sys
    post_counts = {format(i, f"0{n_sys}b"): 0 for i in range(dim_sys)}
    total_postselected = 0

    for bitstring, count in counts.items():
        parts = bitstring.split()
        if len(parts) == 2:
            sys_str, anc_str = parts[0], parts[1]
        elif len(parts) == 1:
            # sys_bits are leading, ancilla is trailing or vice versa
            anc_str = parts[0][-1]
            sys_str = parts[0][:-1]
        else:
            continue

        if anc_str == "1":
            total_postselected += count
            if sys_str in post_counts:
                post_counts[sys_str] += count
            else:
                post_counts[sys_str] = count

    success_prob = total_postselected / float(total_shots) if total_shots > 0 else 0.0

    empirical_probs = np.zeros(dim_sys, dtype=float)
    if total_postselected > 0:
        for i in range(dim_sys):
            b_str = format(i, f"0{n_sys}b")
            empirical_probs[i] = post_counts.get(b_str, 0) / float(total_postselected)

    return {
        "postselected_counts": post_counts,
        "total_postselected": total_postselected,
        "success_probability": success_prob,
        "empirical_probabilities": empirical_probs,
    }
