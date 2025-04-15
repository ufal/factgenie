import re
import unittest

# Throws `KeyError` when the key is not found.
def template_replace(text: str, label_for_data: str, data):
    """
    Args:
        text: The template string.
        label_for_data: How the data is called in the template (e.g. "data" for {data}).
        data: The data to use for replacing.
    
    Replaces in text:
        - "{data}" with `str(data)`.
        - "{data[key]}" with `str(data[key])`. Any depth is supported (e.g. "{data[a][b][c]}" will work).

    Exceptions:
        - TypeError: Raised when using key access (i.e. {data[key]}) on a non-dictionary data.
        - KeyError: Raised when key doesn't exist in a data.

    The method protects against recursion. I.e. if data = "{data}", it will only replace once.
    """
    # Regex matches "{data[...]+}", with at least one [...] key.
    # Raw would look like: r"{data((?:\[[^\[\]\{\}]*\])+?)}".
    regex = f"{{{label_for_data}((?:\\[[^\\[\\]\\{{\\}}]*\\])+?)?}}"

    # Once we replace some part of text, we want to have that replaced text inaccessible to future regex searches, as it could potentially cause infinite recursion. Therefore we keep the variable `processed_chars`, which remembers how many chars we have processed so far. After each replace, it will point to the rightmost part of that replce.
    processed_chars = 0

    while True:
        # Try to find a match.
        s = re.search(regex, text[processed_chars:])
        if s is None:
            break

        # Find the span to be replaced, taking into account the initial cropping.
        l, r = s.span()
        l += processed_chars
        r += processed_chars

        # Case 1: we have something like {data[key1]}.
        if s.group(1) is not None:
            if type(data) is not dict:
                raise TypeError(f"Trying to access dictionary entries in {label_for_data} but it's not a dictionary.")

            # The replacement path in the dictionary.
            keys = s.group(1)[1:-1].split("][")

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


class TestTemplating(unittest.TestCase):
    def test_template_full(self):
        text = "Yes in this we have {data[a][d]} and {data[e]} and also {data[a][b][c]} and {data[num]}."
        data = {'a': {'b': {'c': '<CC>'}, 'd': '<DD>'}, 'e': '<EE>', 'num': 33}
        self.assertEqual(template_replace(text, "data", data), "Yes in this we have <DD> and <EE> and also <CC> and 33.")

    def test_template_non_dict(self):
        text = "Cats and {data}."
        data = "dogs"
        self.assertEqual(template_replace(text, "data", data), "Cats and dogs.")

    def test_extract(self):
        data = {'a': {'b': {'c': '<CC>'}, 'd': '<DD>'}, 'e': '<EE>'}
        path = ['a', 'b', 'c']
        self.assertEqual(extract_data(data, path), '<CC>')
    
    def test_template_no_recursion(self):
        text = "data[a] is '{data[a]}'"
        data = {'a': '{data[a]}'}
        self.assertEqual(template_replace(text, "data", data), "data[a] is '{data[a]}'")

    def test_extract_wrong_key(self):
        data = {'a': {'b': {'c': '<CC>'}, 'd': '<DD>'}, 'e': '<EE>'}
        self.assertRaises(KeyError, lambda: extract_data(data, ['a', 'z']))

    def test_template_wrong_key(self):
        text = "Yes in this we have {data[a][z]} and {data[e]} and also {data[a][b][c]}."
        data = {'a': {'b': {'c': '<CC>'}, 'd': '<DD>'}, 'e': '<EE>'}
        self.assertRaises(KeyError, lambda: template_replace(text, "data", data))


if __name__ == '__main__':
    unittest.main()
