from pathlib import Path

PACKAGE_DIR = Path(__file__).parent
ROOT_DIR = PACKAGE_DIR.parent

TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"
CAMPAIGN_DIR = PACKAGE_DIR / "campaigns"
LLM_EVAL_CONFIG_DIR = PACKAGE_DIR / "config" / "llm-eval"
LLM_GEN_CONFIG_DIR = PACKAGE_DIR / "config" / "llm-gen"
CROWDSOURCING_CONFIG_DIR = PACKAGE_DIR / "config" / "crowdsourcing"

INPUT_DIR = PACKAGE_DIR / "data" / "inputs"
OUTPUT_DIR = PACKAGE_DIR / "data" / "outputs"

DATASET_CONFIG_PATH = PACKAGE_DIR / "data" / "datasets.yml"
RESOURCES_CONFIG_PATH = PACKAGE_DIR / "config" / "resources.yml"

MAIN_CONFIG_PATH = PACKAGE_DIR / "config" / "config.yml"
MAIN_CONFIG_TEMPLATE_PATH = PACKAGE_DIR / "config" / "config_TEMPLATE.yml"
DEFAULT_PROMPTS_CONFIG_PATH = PACKAGE_DIR / "config" / "default_prompts.yml"
PREVIEW_STUDY_ID = "factgenie_preview"
