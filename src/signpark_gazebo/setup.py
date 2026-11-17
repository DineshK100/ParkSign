import os
from glob import glob

from setuptools import setup

package_name = 'signpark_gazebo'


def tree(d):
    """Install every file under directory d into share/<pkg>/d, keeping sub-folders."""
    out = {}
    for f in glob(os.path.join(d, '**', '*'), recursive=True):
        if os.path.isfile(f):
            out.setdefault(os.path.join('share', package_name, os.path.dirname(f)), []).append(f)
    return list(out.items())


setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ] + sum((tree(d) for d in ['config', 'launch', 'signs', 'rviz'] if os.path.isdir(d)), []),
    install_requires=['setuptools'],
    zip_safe=True,
    description='SignPark parking-lot worlds, sign photos, scenarios and simulation launch.',
    license='MIT',
    entry_points={'console_scripts': [
        'make_scenario = signpark_gazebo.world_gen:main',
        'placeholder_signs = signpark_gazebo.placeholder_signs:main',
    ]},
)
