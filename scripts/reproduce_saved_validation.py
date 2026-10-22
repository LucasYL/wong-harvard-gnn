"""Reproduce nine saved GNNs and NN on 64 old validation rows; no training."""
import json

import evaluate_external_synthemol_v1 as original


if __name__ == '__main__':
    result = original.reproduce(original.checked())
    print(json.dumps(result, indent=2))
