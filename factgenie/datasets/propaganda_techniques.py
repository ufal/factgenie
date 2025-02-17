"""
The Propaganda Techniques Corpus (PTC) is a corpus of propagandistic techniques at annotated by 6 pro-annotators in spans.
    The corpus includes 451 articles (350k tokens) from 48 news outlets. 
    The corpus accompanied paper [Fine-Grained Analysis of Propaganda in News Article](https://aclanthology.org/D19-1565/).
    The labels are:
Loaded Language, Name Calling&Labeling, Repetition, Exaggeration&Minimization, Doubt, Appeal to fear-prejudice, Flag-Waving, Causal Oversimplification, Slogans, Appeal to Authority, Black-and-White Fallacy, Thought-terminating Cliches, Whataboutism, Reductio ad Hitlerum, Red Herring, Bandwagon, Obfuscation&Intentional&Vagueness&Confusion, Straw Men
"""

import glob
import json
import logging
import os
from pathlib import Path
from typing import Any, List, Union, Dict
import zipfile

from factgenie.utils import resumable_download
from factgenie.datasets.dataset import Dataset

logger = logging.getLogger("factgenie")

PTC_DATASET_ID = "propaganda-techniques"
PTC_CAMPAIGN_ID = "ptc"

PTC_colors = [
    "#020263",  # Dark Blue
    "#008B8B",  # Dark Cyan
    "#006400",  # Dark Green
    "#B8860B",  # Dark Goldenrod
    "#8B008B",  # Dark Magenta
    "#FF8C00",  # Dark Orange
    "#4B0082",  # Indigo
    "#8B4513",  # Saddle Brown
    "#2F4F4F",  # Dark Slate Gray
    "#556B2F",  # Dark Olive Green
    "#A0522D",  # Sienna
    "#8B0000",  # Dark Red
    "#9932CC",  # Dark Orchid
    "#8f0e42",  # Dark Pink
    "#095700",  # Dark Green
    "#FFD700",  # Gold
    "#FF4500",  # Orange Red
    "#8B4513",  # Saddle Brown
]


# Description extracted from https://www.tanbih.org/persuasion-techniques , normalized label names are taken from the PCT dataset

PTC_span_categories = [
    {
        "name": "Appeal_to_Authority",
        "description": """
    Stating that a claim is true simply because a valid authority or expert on the issue said it was true, without any other supporting evidence offered. We consider the special case in which the reference is not an authority or an expert in this technique, altough it is referred to as Testimonial in literature.
    Example 1: "Richard Dawkins, an evolutionary biologist and perhaps the foremost expert in the field, says that evolution is true. Therefore, it's true."
    Explanation: Richard Dawkins certainly knows about evolution, and he can confidently tell us that it is true, but that doesn't make it true. What makes it true is the preponderance of evidence for the theory.
    Example 2: "According to Serena Williams, our foreign policy is the best on Earth. So we are in the right direction." Details: since there is a chance that any authority can be wrong, it is reasonable to defer to an authority to support a claim, but the authority should not be the only justification to accept the claim, otherwise the Appeal-to-Authority fallacy is committed.
    """,
    },
    {
        "name": "Appeal_to_fear-prejudice",
        "description": """
    Seeking to build support for an idea by instilling anxiety and/or panic in the population towards an alternative.
    In some cases the support is built based on preconceived judgements.
    Example 1: "either we go to war or we will perish" (this is also aBlack and White fallacy))
    Example 2: "we must stop those refugees as they are terrorists"
    """,
    },
    {
        "name": "Bandwagon",
        "description": """
    Attempting to persuade the target audience to join in and take the course of action because "everyone else is taking the same action".
    Example 1: "Would you vote for Clinton as president? 57% say yes"
    Example 2: "90% of citizens support our initiative. You should."
    """,
    },
    {
        "name": "Black-and-White_Fallacy",
        "description": """
    Presenting two alternative options as the only possibilities, when in fact more possibilities exist. As an the extreme case, tell the audience exactly what actions to take, eliminating any other possible choices (Dictatorship).
    Example 1: "You must be a Republican or Democrat. You are not a Democrat. Therefore, you must be a Republican"
    Example 2: "I thought you were a good person, but you weren’t at church today."
    Explanation: The assumption here is that if one doesn't attend chuch, one must be bad. Of course, good people exist who don’t go to church, and good church-going people could have had a really good reason not to be in church.
    Example 3: "There is no alternative to war"
    """,
    },
    {
        "name": "Causal_Oversimplification",
        "description": """
    Assuming a single cause or reason when there are actually multiple causes for an issue.
    It includes transferring blame to one person or group of people without investigating the complexities of the issue
    Example 1: "President Trump has been in office for a month and gas prices have been skyrocketing. The rise in gas prices is because of President Trump."
    Example 2: "The reason New Orleans was hit so hard with the hurricane was because of all the immoral people who live there."
    Explanation: This was an actual argument seen in the months that followed hurricane Katrina. Ignoring the validity of the claims being made, the arguer is blaming a natural disaster on a group of people.
    Example 3: "If France had not have declared war on Germany then world war two would have never happened."
    """,
    },
    {
        "name": "Doubt",
        "desriptiion": """
    Questioning the credibility of someone or something.
    Example 1: "A candidate talks about his opponent and says: Is he ready to be the Mayor?"
    """,
    },
    {
        "name": "Exaggeration,Minimisation",
        "description": """
    Either representing something in an excessive manner: making things larger, better, worse (e.g., "the best of the best", "quality guaranteed") or making something seem less important or smaller than it really is (e.g., saying that an insult was just a joke).
    Example 1: "Democrats bolted as soon as Trump’s speech ended in an apparent effort to signal they can’t even stomach being in the same room as the president "
    Example 2: "We’re going to have unbelievable intelligence"
    Example 3: "I was not fighting with her; we were just playing."
    """,
    },
    {
        "name": "Flag-Waving",
        "description": """
    Playing on strong national feeling (or to any group; e.g., race, gender, political preference) to justify or promote an action or idea
    Example 1: "Patriotism mean no questions" (this is also a slogan)
    Example 2: "Entering this war will make us have a better future in our country."
    """,
    },
    {
        "name": "Loaded_Language",
        "description": """
    Using specific words and phrases with strong emotional implications (either positive or negative) to influence an audience.
    Example 1: [...] a lone lawmaker’s childish shouting.
    Example 2: How stupid and petty things have become in Washington
    """,
    },
    {
        "name": "Name_Calling,Labeling",
        "description": """
    Labeling the object of the propaganda campaign as either something the target audience fears, hates, finds undesirable or loves, praises.
    Example 1: "Republican congressweasels", "Bush the Lesser" (note that lesser does not refer to "the second", but it is pejorative)
    """,
    },
    {
        "name": "Obfuscation,Intentional_Vagueness,Confusion",
        "description": """
    Using words which are deliberately not clear so that the audience may have its own interpretations.
    For example when an unclear phrase with multiple definitions is used within the argument and, therefore, it does not support the conclusion.
    Example 1: "It is a good idea to listen to victims of theft. Therefore if the victims say to have the thief shot, then you should do that."
    Explanation: the definition for "listen to" is equivocated here. In the first case it means listen to their personal account of the experience of being a victim of theft. Empathize with them. In the second case "listen to" means carry out a punishment of their choice.
    """,
    },
    {
        "name": "Red_Herring",
        "description": """
    Introducing irrelevant material to the issue being discussed, so that everyone's attention is diverted away from the points made.
    Example 1: In politics, defending one’s own policies regarding public safety - “I have worked hard to help eliminate criminal activity. What we need is economic growth that can only come from the hands of leadership.”
    Example 2: "You may claim that the death penalty is an ineffective deterrent against crime -- but what about the victims of crime? How do you think surviving family members feel when they see the man who murdered their son kept in prison at their expense? Is it right that they should pay for their son's murderer to be fed and housed?"
    """,
    },
    {
        "name": "Reductio_ad_hitlerum",
        "description": """
    Persuading an audience to disapprove an action or idea by suggesting that the idea is popular with groups hated in contempt by the target audience. It can refer to any person or concept with a negative connotation.
    Example 1: "Do you know who else was doing that ? Hitler!"
    Example 2: "Only one kind of person can think in that way: a communist."
    """,
    },
    {
        "name": "Repetition",
        "description": """
    Repeating the same message over and over again so that the audience will eventually accept it.
    """,
    },
    {
        "name": "Slogans",
        "description": """
    A brief and striking phrase that may include labeling and stereotyping. Slogans tend to act as emotional appeals.
    Example 1: "The more women at war . . . the sooner we win."
    Example 2: "Make America great again!"
    """,
    },
    {
        "name": "Straw_Men",
        "description": """
When an opponent's proposition is substituted with a similar one which is then refuted in place of the original proposition.

Example 1: Zebedee: What is your view on the Christian God?
Mike: I don’t believe in any gods, including the Christian one.
Zebedee: So you think that we are here by accident, and all this design in nature is pure chance, and the universe just created itself?
Mike: You got all that from me stating that I just don’t believe in any gods?

Explanation: Mike made one claim: that he does not believe in any gods. From that, we can deduce a few things, like he is not a theist, he is not a practicing Christian, Catholic, Jew, or a member of any other religion that requires the belief in a god, but we cannot deduce that he believes we are all here by accident, nature is chance, and the universe created itself.
""",
    },
    {
        "name": "Thought-terminating_Cliches",
        "description": """
Words or phrases that discourage critical thought and meaningful discussion about a given topic. They are typically short, generic sentences that offer seemingly simple answers to complex questions or that distract attention away from other lines of thought.


Example 1: "It is what it is"; "It's just common sense"; "You gotta do what you gotta do"; "Nothing is permanent except change"; "Better late than never"; "Mind your own business"; "Nobody's perfect"; "It doesn't matter"; "You can't change human nature."
""",
    },
    {
        "name": "Whataboutism",
        "description": """
A technique that attempts to discredit an opponent's position by charging them with hypocrisy without directly disproving their argument.


Example 1: "A nation deflects criticism of its recent human rights violations by pointing to the history of slavery in the United States."

Example 2: "Qatar spending profusely on Neymar, not fighting terrorism"
""",
    },
]

PTC_span_categories = [{**d, **{"color": c}} for d, c in zip(PTC_span_categories, PTC_colors)]  # add colour


class PropagandaTechniques(Dataset):

    def load_examples(self, split, data_path):
        self._article_id_to_example_idx = {}  # For debugging purposes only
        examples = []
        articles_files = glob.glob(f"{data_path}/{split}/article*.txt")
        for example_idx, f in enumerate(articles_files):
            article_id = str(Path(f).stem)[len("article") :]
            self._article_id_to_example_idx[article_id] = example_idx
            with open(f, "r") as file:
                article = file.read()
                examples.append({"text": article.strip(), "id": article_id})
        return examples

    def render(self, example):
        # # Repeat the article as an input so the user can scroll to other place
        # html = "<div>"
        # html += f"<h3>ID: article{example['id']}.txt </h3>"  # render the full file name
        # html += "<p>"
        # html += example["text"].replace("\\n", "<br>")
        # html += "</p>"
        # html += "</div>"
        # return html
        return None  # for None it shows just the output

    @classmethod
    def download(
        cls,
        dataset_id,
        data_download_dir,
        out_download_dir,
        annotation_download_dir,
        splits,
        outputs,
        dataset_config,
        **kwargs,
    ):
        assert dataset_id == PTC_DATASET_ID, f"Dataset ID {dataset_id} does not match {PTC_DATASET_ID}"
        link = dataset_config["data-link"]
        logger.info(f"Downloading dataset {dataset_id} from {link}")
        resumable_download(url=link, filename=f"{data_download_dir}/{dataset_id}.zip", force_download=False)
        logger.info(f"Downloaded {dataset_id}")

        with zipfile.ZipFile(f"{data_download_dir}/{dataset_id}.zip", "r") as zip_ref:
            zip_ref.extractall(data_download_dir)
        os.remove(f"{data_download_dir}/{dataset_id}.zip")
        # symlink the original data splits in protchn_corpus_eval to the splits in the data_download_dir
        for split in splits:
            if not os.path.exists(f"{data_download_dir}/{split}"):
                os.symlink(f"{data_download_dir}/protechn_corpus_eval/{split}", f"{data_download_dir}/{split}")

        # # we added the categories manually
        # annotation_span_categories_path = f"{data_download_dir}/protechn_corpus_eval/propaganda-techniques-names.txt"
        # with open(annotation_span_categories_path) as r:
        #     annotation_span_categories = r.readlines()

        # save annotations
        annotation_jsonl_parent = annotation_download_dir / PTC_CAMPAIGN_ID / "files"
        annotation_jsonl_parent.mkdir(parents=True, exist_ok=True)

        short_categories = [{"name": c["name"], "color": c["color"], "description": ""} for c in PTC_span_categories]
        categories_names = [c["name"] for c in short_categories]

        # save outputs
        outputs_jsonl_parent = out_download_dir / dataset_id
        outputs_jsonl_parent.mkdir(parents=True, exist_ok=True)

        db_csv_dummy_csv = annotation_download_dir / PTC_CAMPAIGN_ID / "db.csv"
        with open(db_csv_dummy_csv, "wt") as dbw:
            dbw.write("dataset,split,example_idx,setup_id,batch_idx,annotator_id,status,start,end\n")
            for split in splits:

                with (
                    open(outputs_jsonl_parent / f"{split}.jsonl", "wt") as outputw,
                    open(annotation_jsonl_parent / f"{split}.jsonl", "wt") as annotationw,
                ):
                    article_id_to_example_idx = {}
                    articles_files = glob.glob(f"{data_download_dir}/{split}/article*.txt")

                    for example_idx, f in enumerate(articles_files):
                        dbw.write(f"{dataset_id},{split},{example_idx},{PTC_CAMPAIGN_ID},{example_idx},,finished,,\n")
                        article_id = str(Path(f).stem)[len("article") :]
                        article_id_to_example_idx[article_id] = example_idx

                        with open(f, "r") as file:
                            article_txt = file.read().strip()
                        article_entry = {
                            "dataset": dataset_id,
                            "split": split,
                            "setup_id": dataset_id,
                            "example_idx": example_idx,
                            "metadata": {"article": article_id},
                            "output": article_txt,
                        }
                        outputw.write(json.dumps(article_entry) + "\n")

                        annotation_file = (
                            f"{data_download_dir}/protechn_corpus_eval/{split}/article{article_id}.labels.tsv"
                        )
                        annotationw.write(
                            json.dumps(
                                {
                                    "dataset": PTC_DATASET_ID,
                                    "split": split,
                                    "setup_id": PTC_DATASET_ID,
                                    "example_idx": example_idx,
                                    "metadata": {
                                        "annotation_span_categories": short_categories,
                                        "annotator_id": "idk",
                                        "annotator_group": 0,
                                        "campaign_id": PTC_CAMPAIGN_ID,
                                        "article": article_id,  # original dataset id
                                    },
                                    "annotations": cls._load_example_annotations(
                                        annotation_file, article_txt, article_id, categories_names
                                    ),
                                }
                            )
                            + "\n"
                        )

        # save metadata
        metadata_json = annotation_download_dir / PTC_CAMPAIGN_ID / "metadata.json"

        metadata = {
            "id": PTC_CAMPAIGN_ID,
            "mode": "external",
            "config": {
                "annotation_span_categories": PTC_span_categories,
                "flags": [],
                "options": [],
                "sliders": [],
                "text_fields": [],
            },
            "created": "2024-03-25 00:00:00",
        }
        with open(metadata_json, "wt") as w:
            w.write(json.dumps(metadata, indent=2))

    @classmethod
    def _load_example_annotations(
        cls, annotation_file: Union[str, Path], article: str, article_id: str, categories_names: List[str]
    ):
        annotations = []
        with open(annotation_file, "r") as file:
            for annotation in file.readlines():
                _article_id, category, start_idx, end_idx = annotation.strip().split("\t")
                assert (
                    _article_id == article_id
                ), f"Article ID mismatch: {_article_id} != {article_id} in {annotation_file}"
                start_idx, end_idx = int(start_idx), int(end_idx)  # char indices
                try:
                    type_idx = categories_names.index(category)
                except ValueError:
                    logger.warning(f"Category {category} not found in categories_names")
                    continue

                annotation_d = {
                    "reason": "",
                    "text": article[start_idx:end_idx],
                    "type": type_idx,
                    "start": start_idx,
                }
                annotations.append(annotation_d)

        annotations = sorted(annotations, key=lambda x: x["start"])
        return annotations


if __name__ == "__main__":
    # testing download
    from factgenie.bin.run import create_app
    import factgenie.workflows as workflows

    app = create_app()

    workflows.download_dataset(app, "propaganda-techniques")
