# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections import defaultdict

from inspect_ai.scorer import Metric, SampleScore, metric, stderr


@metric
def equal_weight_group_stderr(group_key: str) -> Metric:
    """
    Compute per-group standard errors and the standard error of their equal-weight mean.

    The aggregate assumes independent groups.

    Args:
        group_key (`str`):
            Sample metadata key identifying the group.
    """

    def grouped_stderr(scores: list[SampleScore]) -> dict[str, float]:
        groups = defaultdict(list)
        for score in scores:
            groups[str(score.sample_metadata[group_key])].append(score)

        standard_error = stderr()
        result = {f"{group}_stderr": standard_error(group_scores) for group, group_scores in groups.items()}
        result["stderr"] = math.sqrt(sum(value**2 for value in result.values())) / len(result)
        return result

    return grouped_stderr
