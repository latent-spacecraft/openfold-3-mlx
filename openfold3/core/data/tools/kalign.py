# Copyright 2025 AlQuraishi Laboratory
# Copyright 2025 Geoffrey Taghon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import shutil
import subprocess
from functools import lru_cache


@lru_cache(maxsize=512)
def run_kalign(
    a3m_string: str,
) -> str:
    """Runs Kalign on the provided A3M string and returns the aligned sequences.

    Args:
        a3m_string (str):
            A3M formatted string containing the sequences to be aligned. In the template
            pipeline, the first sequence is the query, and the rest are templates
            sequences to be realigned to it from hmmsearch.

    Raises:
        RuntimeError:
            If Kalign is not available.
        subprocess.CalledProcessError:
            If the Kalign command fails.

    Returns:
        str:
            The aligned sequences in A3M format as a string.
    """
    kalign_available = shutil.which("kalign") is not None

    if not kalign_available:
        raise RuntimeError(
            "Kalign is not available. Please install it and ensure it is in your PATH."
        )

    try:
        result = subprocess.run(
            ["kalign", "--format", "fasta", "--type", "protein"],
            input=a3m_string, capture_output=True, text=True, check=True
        )

        # The resulting MSA is stored in the variable
        alignment_result = result.stdout

        # Filter out kalign header/log information to extract only FASTA sequences
        lines = alignment_result.split('\n')
        fasta_lines = []
        found_fasta_start = False

        for line in lines:
            # Skip header/copyright/log lines
            if (line.startswith('Kalign') or
                line.startswith('Copyright') or
                line.startswith('This program') or
                line.startswith('This is free software') or
                line.startswith('Please cite:') or
                line.startswith('  Lassmann') or
                line.startswith('  Bioinformatics') or
                line.startswith('  https://') or
                line.startswith('[') or  # Log lines like "[2025-11-13"
                line.strip() == '' and not found_fasta_start):
                continue
            elif line.startswith('>'):
                # Found first FASTA header
                found_fasta_start = True
                fasta_lines.append(line)
            elif found_fasta_start:
                fasta_lines.append(line)

        alignment_result = '\n'.join(fasta_lines)

    except subprocess.CalledProcessError as e:
        print(f"Kalign command failed:\n{e.stderr}")
        raise RuntimeError(f"Kalign alignment failed: {e.stderr}") from e

    return alignment_result
