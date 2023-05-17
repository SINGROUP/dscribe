# -*- coding: utf-8 -*-
"""Copyright 2019 DScribe developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np


def is1d(array, dtype=np.integer):
    try:
        for i in array:
            if not np.issubdtype(type(i), dtype):
                return False
    except Exception:
        return False
    return True


def is2d(array, dtype=np.integer):
    try:
        for i in array:
            for j in i:
                if not np.issubdtype(type(j), dtype):
                    return False
    except Exception:
        return False
    return True
