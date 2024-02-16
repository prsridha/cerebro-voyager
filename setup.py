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
# ==============================================================================

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
# ==============================================================================

from distutils.core import setup
import os
this = os.path.dirname(__file__)

with open(os.path.join(this, "requirements.txt"), "r") as f:
    requirements = [_ for _ in [_.strip("\r\n ")
                                for _ in f.readlines()] if _ is not None]

packages = ["cerebro"]
assert packages

# read version from the package file.
version_str = '1.0.0'
with (open(os.path.join(this, 'cerebro/__init__.py'), "r")) as f:
    line = [_ for _ in [_.strip("\r\n ")
                                for _ in f.readlines()] if _.startswith("__version__")]
    if len(line) > 0:
        version_str = line[0].split('=')[1].strip('" ')

setup(
    name='cerebro',
    version=version_str,
    description="Resource-efficient Deep Learning Model Selection on Data Systems",
    long_description_content_type='text/markdown',
    license='Apache License 2.0',
    author='adalabs',
    author_email='prsridha@ucsd.edu',
    url='https://adalabucsd.github.io/cerebro.html',
    packages=packages,
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False
)