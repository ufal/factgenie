#!/usr/bin/env python3

import re
import unittest


def find_all_template_keys(text: str) -> list[str]:
    """
    Returns:
        A list of all '{...}' template keywords in the template.
    """
    regex = f"{{([\\w-]+)(?:(?:\\[[^\\[\\]\\{{\\}}]*\\])+?)?}}"
    results = re.findall(regex, text)
    return results


def template_replace(text: str, keyword_dict: dict):
    """
    Args:
        text: The template string.
        keyword_dict: A dictionary defining template keys and corresponding values. See the example below.

    Example: `keyword_dict = { "data": data }` will replace:
        - "{data}" with `str(data)`.
        - "{data[key]}" with `str(data[key])`. Any depth is supported (e.g. "{data[a][b][c]}" will work).

    Exceptions:
        - TypeError: Raised when using key access (i.e. {data[key]}) on a non-dictionary data.
        - KeyError: Raised when key doesn't exist in the data.

    The method protects against recursion. I.e. if data = "{data}", it will only replace once.

    "{unknown_keywords}" in text will not be detected.
    """
    # Regex matches "{data[...]+}", with at least one [...] key.
    # Where 'data' is one of the keyword_dict keys.
    # Raw would look like: r"{data((?:\[[^\[\]\{\}]*\])+?)}".
    regex = f"{{({'|'.join(keyword_dict.keys())})((?:\\[[^\\[\\]\\{{\\}}]*\\])+?)?}}"

    # Once we replace some part of text, we want to have that replaced text inaccessible to future regex searches, as it could potentially cause infinite recursion. Therefore we keep the variable `processed_chars`, which remembers how many chars we have processed so far. After each replace, it will point to the rightmost part of that replce.
    processed_chars = 0

    while True:
        # Try to find a match.
        s = re.search(regex, text[processed_chars:])
        if s is None:
            break

        label_for_data = s.group(1)
        data = keyword_dict[label_for_data]

        # Find the span to be replaced, taking into account the initial cropping.
        l, r = s.span()
        l += processed_chars
        r += processed_chars

        # Case 1: we have something like {data[key1]}.
        if s.group(2) is not None:
            if type(data) is not dict:
                raise TypeError(f"Trying to access dictionary entries in {label_for_data} but it's not a dictionary.")

            # The replacement path in the dictionary.
            keys = s.group(2)[1:-1].split("][")

            # Replace.
            try:
                replace_with = str(extract_data(data, keys))
            except KeyError:
                raise KeyError(f"Could not find path {s.group(0)} in {label_for_data}.")

        # Case 2: we have just {data}.
        else:
            replace_with = str(data)

        text = text[:l] + replace_with + text[r:]
        processed_chars = l + len(replace_with)

    return text


def extract_data(data: dict, keys: list[str]):
    """
    `extract_data(data, ['a', 'b'])` either returns data['a']['b'] or throws a KeyError when not found.
    """
    key = keys[0]
    tail = keys[1:]

    if key not in data:
        raise KeyError()

    value = data[key]

    if len(tail) == 0:
        return value
    elif type(value) is dict:
        return extract_data(value, tail)
    else:
        raise KeyError()


def iter_sentences(text: str):
    # This regex:
    #  - '.' and a negative lookahead
    #    - Can't be followed by another numer, comma, colon, or spaces* lowercase.
    #      This is needed for decimals (3.5) and abbreviations (e.g. this).
    #    - Can be followed by an optional \".
    #  - '?' or '!' followed by an optional \".
    #  - Extra chunking shouldn't hurt once I show it preceding context.
    # punc_regex = "\\.(?![0-9]|,|:|\\s*[a-z])\"?|\\?\"?|!\"?(?!\\.|\\?|!|,)"
    punc_regex = '(?:\\.|\\?|\\!)(?![0-9]|,|:|-|\\s*[a-z]|\\.|\\?|!)"?\\s?'
    processed_chars = 0

    while True:
        s = re.search(punc_regex, text[processed_chars:])
        if s is None:
            break

        _, r = s.span()
        yield text[processed_chars : processed_chars + r].strip()
        processed_chars += r

    if processed_chars < len(text):
        yield text[processed_chars:].strip()


def join_outer_lists(texts: list[str], json_header: str | None = None, min_length: int = 7):
    """
    Finds the outer list in each text, delimited by the outer-most '[' and ']', and joins them together.

    if json_header is specified, the result will be formatted like this:
    ```json
    {
        {json_header}: [
            ...,  # from texts[0]
            ...,  # from texts[1]
            ...   # ...
        ]
    }
    ```
    otherwise, the result will look like this:
    ```json
    [
        ...,  # from texts[0]
        ...,  # from texts[1]
        ...   # ...
    ]
    ```

    If no '[' ']' pair is found, or if the length of the text inside is less than min_length, the text will be skipped.
    """
    sections = []
    for a in texts:
        left = a.find("[")
        right = a.rfind("]")

        # If either `find` or `rfind` failed and returned -1, skip.
        if left < 0 or right < 0:
            continue

        # A random threshold to avoid empty annotations; no annotation should be this short.
        # If the list is too short (e.g. empty), skip.
        if right - left < min_length:
            continue

        sections.append(a[left + 1 : right])

    if json_header is not None:
        # Formatting:
        # {
        #     {json_header}: [
        #         ...,
        #         ...
        #     ]
        # }
        joint = f'{{\n\t"{json_header}": [\n\t\t' + ",\n\t\t".join(sections) + "\n\t]\n}"
    else:
        # Formatting:
        # [
        #     ...,
        #     ...
        # ]
        joint = "[\n\t" + ",\n\t".join(sections) + "\n]"

    return joint


# ――――――――――――――――――――――――――――――――――― TESTS ―――――――――――――――――――――――――――――――――――


class TestTemplating(unittest.TestCase):
    def test_template_full(self):
        text = "Yes in this we have {data[a][d]} and {data[e]} and also {data[a][b][c]} and {data[num]}."
        data = {"a": {"b": {"c": "<CC>"}, "d": "<DD>"}, "e": "<EE>", "num": 33}
        keyword_dict = {"data": data}
        self.assertEqual(
            template_replace(text, keyword_dict),
            "Yes in this we have <DD> and <EE> and also <CC> and 33.",
        )

    def test_template_non_dict(self):
        text = "Cats and {data}."
        data = "dogs"
        keyword_dict = {"data": data}
        self.assertEqual(template_replace(text, keyword_dict), "Cats and dogs.")

    def test_extract(self):
        data = {"a": {"b": {"c": "<CC>"}, "d": "<DD>"}, "e": "<EE>"}
        path = ["a", "b", "c"]
        self.assertEqual(extract_data(data, path), "<CC>")

    def test_template_no_recursion(self):
        text = "data[a] is '{data[a]}'"
        data = {"a": "{data[a]}"}
        keyword_dict = {"data": data}
        self.assertEqual(template_replace(text, keyword_dict), "data[a] is '{data[a]}'")

    def test_extract_wrong_key(self):
        data = {"a": {"b": {"c": "<CC>"}, "d": "<DD>"}, "e": "<EE>"}
        self.assertRaises(KeyError, lambda: extract_data(data, ["a", "z"]))

    def test_template_wrong_key(self):
        text = "Yes in this we have {data[a][z]} and {data[e]} and also {data[a][b][c]}."
        data = {"a": {"b": {"c": "<CC>"}, "d": "<DD>"}, "e": "<EE>"}
        keyword_dict = {"data": data}
        self.assertRaises(KeyError, lambda: template_replace(text, keyword_dict))


class TestSentenceSplit(unittest.TestCase):
    def test_sentence_split(self):
        text = "Hey. What is your name? Mine is Filip!!?! And my favorite number is 3.14159265358979323846264338"
        expected = [
            "Hey.",
            "What is your name?",
            "Mine is Filip!!?!",
            "And my favorite number is 3.14159265358979323846264338",
        ]
        actual = list(iter_sentences(text))
        self.assertListEqual(expected, actual)


class TestJoinOuterLists(unittest.TestCase):
    def test_no_header(self):
        input = ["[{'a': 'a', 'b': [{'c': 'c'}, {'d': 'd'}] }]", "", "[]", "[{'a': 'a2'}]"]
        result = join_outer_lists(input)
        expected = "[\n\t{'a': 'a', 'b': [{'c': 'c'}, {'d': 'd'}] },\n\t{'a': 'a2'}\n]"

        # Remove whitespaces as we don't want to check exact formatting.
        result_no_whitespace = re.sub(r"\s*", "", result)
        expected_no_whitespace = re.sub(r"\s*", "", expected)

        self.assertEqual(result_no_whitespace, expected_no_whitespace)

    def test_header(self):
        input = ["[{'a': 'a', 'b': [{'c': 'c'}, {'d': 'd'}] }]", "", "[]", "[{'a': 'a2'}]"]
        result = join_outer_lists(input, json_header="annotations")
        expected = "{\n\t\"annotations\": [\n\t\t{'a': 'a', 'b': [{'c': 'c'}, {'d': 'd'}] },\n\t\t{'a': 'a2'}\n\t]\n}"

        # Remove whitespaces as we don't want to check exact formatting.
        result_no_whitespace = re.sub(r"\s*", "", result)
        expected_no_whitespace = re.sub(r"\s*", "", expected)

        self.assertEqual(result_no_whitespace, expected_no_whitespace)

    def test_invalid_entries(self):
        input = [
            "[{'a': 'a', 'b': [{'c': 'c'}, {'d': 'd'}] }]",
            "",
            "sdfsd]",
            "dsfseese[",
            "ahojklsejfasoiuehdf { sfefes fse [3fdf]}",
            "[{'a': 'a2'}]",
        ]
        result = join_outer_lists(input)
        expected = "[\n\t{'a': 'a', 'b': [{'c': 'c'}, {'d': 'd'}] },\n\t{'a': 'a2'}\n]"

        # Remove whitespaces as we don't want to check exact formatting.
        result_no_whitespace = re.sub(r"\s*", "", result)
        expected_no_whitespace = re.sub(r"\s*", "", expected)

        self.assertEqual(result_no_whitespace, expected_no_whitespace)


if __name__ == "__main__":
    unittest.main()
