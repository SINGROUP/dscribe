import re
import os

coding_text = "# -*- coding: utf-8 -*-\n"
coding_re = "^"+re.escape(coding_text)
copyright_re = ("\"\"\"\nCopyright 2019 DScribe developers\n")
copyright_text = """\"\"\"Copyright 2019 DScribe developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
\"\"\"
"""


def match_func(filename):
    return filename.endswith(".py")


def crawl():
    # folders = ["./dscribe/utils"]
    # folders = ["./dscribe/kernels"]
    # folders = ["./dscribe/core"]
    folders = ["./dscribe/descriptors"]
    for folder in folders:
        for path, dirs, files in os.walk(folder):
            for f in files:
                filename = os.path.join(path, f)
                if match_func(f):

                    with open(filename, "r+") as fh:
                        text = fh.read()

                        pos = 0

                        # Try to find coding.
                        match = re.search(coding_re, text)
                        if match is None:
                            text = coding_text + text
                        else:
                            pos = match.end()

                        # Try to find copyright. If not present add after coding
                        match = re.search(copyright_re, text)
                        if match is None:
                            text = text[0:pos] + copyright_text + text[pos:]

                        # Write new contents
                        fh.seek(0)
                        fh.write(text)
                        fh.truncate()

crawl()
