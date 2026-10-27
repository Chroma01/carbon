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

import os


def init_trackio_training(training_args):
    """
    Helper function for setting up Trackio logging.
    """
    if training_args.project is not None:
        os.environ["TRACKIO_PROJECT"] = training_args.project
    if training_args.trackio_space_id is not None:
        os.environ["TRACKIO_SPACE_ID"] = training_args.trackio_space_id
