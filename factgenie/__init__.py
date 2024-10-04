from pathlib import Path

PACKAGE_DIR = Path(__file__).parent
ROOT_DIR = PACKAGE_DIR.parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"
ANNOTATIONS_DIR = PACKAGE_DIR / "annotations"
GENERATIONS_DIR = PACKAGE_DIR / "generations"
LLM_EVAL_CONFIG_DIR = PACKAGE_DIR / "config" / "llm-eval"
LLM_GEN_CONFIG_DIR = PACKAGE_DIR / "config" / "llm-gen"
CROWDSOURCING_CONFIG_DIR = PACKAGE_DIR / "config" / "crowdsourcing"

DATA_DIR = PACKAGE_DIR / "data"
OUTPUT_DIR = PACKAGE_DIR / "outputs"

DATASET_CONFIG_PATH = PACKAGE_DIR / "config" / "datasets.yml"
DATASET_LOCAL_CONFIG_PATH = PACKAGE_DIR / "config" / "datasets_local.yml"

OLD_DATASET_CONFIG_PATH = PACKAGE_DIR / "loaders" / "datasets.yml"
OLD_MAIN_CONFIG_PATH = PACKAGE_DIR / "config.yml"

MAIN_CONFIG_PATH = PACKAGE_DIR / "config" / "config.yml"
if not MAIN_CONFIG_PATH.exists():
    raise ValueError(
        f"Invalid path to config.yml {MAIN_CONFIG_PATH=}. "
        "Please rename config_TEMPLATE.yml to config.yml. "
        "Change the password, update the host prefix, etc."
    )

PREVIEW_STUDY_ID = "factgenie_preview"
