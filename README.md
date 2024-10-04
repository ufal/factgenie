<div align="center">
<img src="factgenie/static/img/factgenie_whitebg.png" width=80px" alt="logo" />

<h1> factgenie </h1>

![GitHub](https://img.shields.io/github/license/kasnerz/factgenie)
![GitHub issues](https://img.shields.io/github/issues/kasnerz/factgenie)
[![arXiv](https://img.shields.io/badge/arXiv-2407.17863-0175ac.svg)](https://arxiv.org/abs/2407.17863)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Github stars](https://img.shields.io/github/stars/kasnerz/factgenie?style=social)
<!-- ![PyPI](https://img.shields.io/pypi/v/factgenie) -->
<!-- ![PyPI downloads](https://img.shields.io/pypi/dm/factgenie) -->

Annotate LLM outputs with a lightweight, self-hosted web application 🌈

![factgenie](https://github.com/user-attachments/assets/1d074588-ada1-4974-a42a-0d2195c65283)

</div>

## 📢  News
- **24/09/2024** - We introduced a brand new factgenie logo!
- **19/09/2024** - On the Analytics page, you can now see detailed statistics about annotations and compute inter-annotator agreement 📈
- **16/09/2024** - You can now collect extra inputs from the annotators for each example using sliders and selectboxes. 
- **16/09/2024** - We added an option to generate outputs for the inputs with LLMs directly within factgenie! 🦾
- **10/09/2024** - We improved UX for the annotation and browsing interfaces. See the summary of changes [here](https://github.com/ufal/factgenie/pull/71).

## 👉️ How can factgenie help you?
Outputs from large language models (LLMs) may contain errors: semantic, factual, and lexical. 

With factgenie, you can have the error spans annotated:
- From LLMs through an API.
- From humans through a crowdsourcing service.

Factgenie can provide you:
1. **A user-friendly website** for collecting annotations from human crowdworkers.
2. **API calls** for collecting equivalent annotations from LLM-based evaluators.
3. **A visualization interface** for visualizing the data and inspecting the annotated outputs.

---
*What does factgenie **not help with** is collecting the data (we assume that you already have these), starting the crowdsourcing campaign (for that, you need to use a service such as [Prolific.com](https://prolific.com)) or running the LLM evaluators (for that, you need a local framework such as [Ollama](https://ollama.com) or a proprietary API).*

## 🏃 Quickstart
Make sure you have Python 3 installed (the project is tested with Python 3.10).

After cloning the repository, the following commands install the package and start the web server:
```
pip install -e .
factgenie run --host=127.0.0.1 --port 5000
```

## 💡 Usage guide


See the following **wiki pages** that that will guide you through various use-cases of factgenie:

| Topic                                                                  | Description                                        |
| ---------------------------------------------------------------------- | -------------------------------------------------- |
| 🔧 [Setup](../../wiki/01-Setup)                                         | How to install factgenie.                          |
| 🗂️ [Data Management](../../wiki/02-Data-Management)                     | How to manage datasets and model outputs.          |
| 🤖 [LLM Annotations](../../wiki/03-LLM-Annotations)                     | How to annotate outputs using LLMs.                |
| 👥 [Crowdsourcing Annotations](../../wiki/04-Crowdsourcing-Annotations) | How to annotate outputs using human crowdworkers.  |
| ✍️  [Generating Outputs](../../wiki/05-Generating-Outputs)              | How to generate outputs using LLMs.                |
| 📊 [Analyzing Annotations](../../wiki/06-Analyzing-Annotations)         | How to obtain statistics on collected annotations. |
| 🌱 [Contributing](../../wiki/07-Contributing)                           | How to contribute to factgenie.                    |

## 🔥 Tutorials
We also provide a set of hands-on tutorials, showing how to employ factgenie on the [the dataset from the Shared Task in Evaluating Semantic Accuracy](https://github.com/ehudreiter/accuracySharedTask):

| Tutorial                                                                                                                       | Description                                                                                      |
| ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| [🏀 #1: Importing a custom dataset](../../wiki/00-Tutorials#-tutorial-1-importing-a-custom-dataset)                             | Loading the basketball statistics and model-generated basketball reports into the web interface. |
| [💬 #2: Generating outputs](../../wiki/00-Tutorials#-tutorial-2-generating-outputs)                                             | Using Llama 3.1 with Ollama for generating basketball reports.                                   |
| [📊 #3: Customizing data visualization](../../wiki/00-Tutorials#-tutorial-3-customizing-data-visualization)                     | Manually creating a custom dataset class for better data visualization.                          |
| [🤖 #4: Annotating outputs with an LLM](../../wiki/00-Tutorials#-tutorial-4-annotating-outputs-with-an-llm)                     | Using GPT-4o for annotating errors in the basketball reports.                                    |
| [👨‍💼 #5: Annotating outputs with human annotators](../../wiki/00-Tutorials#-tutorial-5-annotating-outputs-with-human-annotators) | Using human annotators for annotating errors in the basketball reports.                          |


## 💬 Cite us

[Our paper](https://aclanthology.org/2024.inlg-demos.5/) was published at INLG 2024 System Demonstrations!

You can also find the paper on [arXiv](https://arxiv.org/abs/2407.17863).

For citing us, please use the following BibTeX entry:
```bibtex
@inproceedings{kasner2024factgenie,
    title = "factgenie: A Framework for Span-based Evaluation of Generated Texts",
    author = "Kasner, Zden{\v{e}}k  and
      Platek, Ondrej  and
      Schmidtova, Patricia  and
      Balloccu, Simone  and
      Dusek, Ondrej",
    editor = "Mahamood, Saad  and
      Minh, Nguyen Le  and
      Ippolito, Daphne",
    booktitle = "Proceedings of the 17th International Natural Language Generation Conference: System Demonstrations",
    year = "2024",
    address = "Tokyo, Japan",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.inlg-demos.5",
    pages = "13--15",
}
```

## Acknowledgements
This work was co-funded by the European Union (ERC, NG-NLG, 101039303).

<img src="img/LOGO_ERC-FLAG_FP.png" alt="erc-logo" height="150"/> 
