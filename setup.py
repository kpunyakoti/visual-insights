import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = '0.0.1'

REPO_NAME = "visual-insights"
DESCRIPTION = 'Visual Insights project'
SRC_REPO = "visualInsights"
AUTHOR = 'CS6242-team165'

setuptools.setup(
    name= SRC_REPO,
    version = __version__,
    author = AUTHOR,
    description = DESCRIPTION,
    long_description = long_description,
    url = f"https://github.com/kpunyakoti/{REPO_NAME}",
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src")
)
