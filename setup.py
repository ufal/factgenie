from pathlib import Path
from setuptools import find_packages, setup
from setuptools.command.install import install
import shutil

project_root = Path(__file__).parent
install_requires = [
    "coloredlogs>=15.0.1",
    "datasets>=2.9.0",
    "Flask-SSE>=1.0.0",
    "Flask>=2.2.2",
    "flask_apscheduler>=1.12.4",
    "json2table>=1.1.5",
    "lxml>=5.3.0",
    "Markdown>=3.7.0",
    "natsort>=8.4.0",
    "python-slugify>=8.0.4",
    "PyYAML>=6.0.2",
    "requests>=2.32.3",
    "scipy>=1.14.1",
    "tinyhtml>=1.2.0",
    "litellm>=1.70.0",
    "pydantic>=2.10.6",
    "google-api-python-client>=2.16.0",
    "google-cloud-aiplatform>=1.38",
    "pygamma-agreement==0.5.9",
    "matplotlib>=3.10.0",
    "tabulate>=0.9.0",
    "tenacity>=9.0.0",
]

setup(
    name="factgenie",
    version="1.1.0",
    python_requires=">=3.9",
    description="Lightweight self-hosted span annotation tool",
    # contributors as on GitHub
    author="Zdenek Kasner, Ondrej Platek, Patricia Schmidtova, Dave Howcroft, Ondrej Dusek",
    author_email="kasner@ufal.mff.cuni.cz",
    long_description=(project_root / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/ufal/factgenie",
    license="MIT",
    packages=find_packages(exclude=["test", "test.*"]),
    package_data={
        "factgenie": [
            "config/default_prompts.yml",
            "config/resources.yml",
            "config/config_TEMPLATE.yml",
            "config/**/example-*.yaml",
            "data/datasets_TEMPLATE.yml",
            "static/**/*",
            "templates/**/*",
        ],
    },
    # data_files=[("factgenie", ["factgenie/config.yml"])],
    # include_package_data=True,
    entry_points={
        "console_scripts": [
            "factgenie=factgenie.bin.run:run",
        ],
        "flask.commands": [
            "create_llm_campaign=factgenie.bin.run:create_llm_campaign",
            "run_llm_campaign=factgenie.bin.run:run_llm_campaign",
            "list=factgenie.bin.run:list_data",
            "info=factgenie.bin.run:info",
        ],
    },
    install_requires=install_requires,
    extras_require={
        "dev": [
            "wheel>=0.44.0",
            # Set exact version of black formatter to avoid merge conflicts due to different setup.
            # See also pyproject.toml and setup of line length (to 120 characters)
            "ipdb",
            "black==24.10.0",
            "isort==5.13.2",
        ],
        "test": [
            "pytest>=8.3.3",
        ],
        "deploy": [
            "gunicorn>=23.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
)
